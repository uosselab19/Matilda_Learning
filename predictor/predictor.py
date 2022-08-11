from network import networks
# import kaolin related
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection, generate_transformation_matrix
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices

# import
import os
import torch
import time
import cv2

import aspose.threed as a3d
from PIL import Image
import numpy as np

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt
import posixpath

from PerceptualSimilarity.models import dist_model

### get_model ###
def get_predictor_model(template_path, resume_path, image_size):
    assert os.path.exists(resume_path)

    diffRender = DiffRender(filename_obj=template_path, image_size=image_size)

    # netE: 3D attribute encoder: Light, Shape, and Texture
    netE = networks.AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, nc=3, nk=5, nf=32)
    netE = netE.cuda()
    netE.eval()

    print("=> loading checkpoint '{}'".format(resume_path))
    # Map model to be loaded to specified single gpu.
    checkpoint = torch.load(resume_path)
    netE.load_state_dict(checkpoint['netE'])
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    return netE, diffRender

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

#####################################################################
# ------------------------- PerceptualLoss ------------------------ #
#####################################################################

class PerceptualLoss(object):
    def __init__(self, model='net', net='alex', use_gpu=True):
        print('Setting up Perceptual loss..')
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        print('Done')

    def __call__(self, pred, target, normalize=True):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        dist = self.model.forward_pair(target, pred)

        return dist

class PerceptualTextureLoss(object):
    def __init__(self):
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        """

        # Only use mask_gt..
        dist = self.perceptual_loss(img_pred, img_gt)
        return dist.mean()

#####################################################################
# ----------------------------- DIB_R ----------------------------- #
#####################################################################

def recenter_vertices(vertices, vertice_shift):
    """Recenter vertices on vertice_shift for better optimization"""
    vertices_min = vertices.min(dim=1, keepdim=True)[0]
    vertices_max = vertices.max(dim=1, keepdim=True)[0]
    vertices_mid = (vertices_min + vertices_max) / 2
    vertices = vertices - vertices_mid + vertice_shift
    return vertices

class DiffRender(object):
    # Hyperparameters
    num_epoch = 210
    batch_size = 1

    laplacian_weight = 0.3  # ring -> 0.3 , shirts -> 0.01
    image_weight = 0.1
    mask_weight = 1.
    perceptual_wight = 0.01
    mov_weight = 0.5

    texture_lr = 3e-2
    vertice_lr = 5e-2
    camera_lr = 5e-4

    scheduler_step_size = 40
    scheduler_gamma = 0.5

    thumb_nail_id = 0
    test_batch_size = 1

    def __init__(self, filename_obj, image_size):
        self.image_size = image_size
        # camera projection matrix
        camera_fovy = np.arctan(1.0 / 2.5) * 2
        self.cam_proj = generate_perspective_projection(camera_fovy, ratio=image_size / image_size)
        self.texture_loss = PerceptualTextureLoss()

        mesh = kal.io.obj.import_mesh(filename_obj, with_materials=True)

        # get vertices_init
        vertices = mesh.vertices.cuda()
        vertices.requires_grad = False
        vertices_max = vertices.max(0, True)[0]
        vertices_min = vertices.min(0, True)[0]
        vertices = (vertices - vertices_min) / (vertices_max - vertices_min)
        vertices_init = vertices * 2.0 - 1.0  # (1, V, 3)

        # get face_uvs
        faces = mesh.faces.cuda()
        uvs = mesh.uvs.cuda().unsqueeze(0)
        face_uvs_idx = mesh.face_uvs_idx.cuda()
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
        face_uvs.requires_grad = False

        self.num_faces = faces.shape[0]
        self.num_vertices = vertices_init.shape[0]

        # flip index
        vertex_center_flip = vertices_init.clone()
        vertex_center_flip[:, 2] *= -1
        self.flip_index = torch.cdist(vertices_init, vertex_center_flip).min(1)[1]

        ## Set up auxiliary laplacian matrix for the laplacian loss
        vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(self.num_vertices, faces)

        # This is the center of the optimized mesh, separating it as a learnable parameter helps the optimization.
        self.vertice_shift = torch.zeros((3,), dtype=torch.float, device='cuda',
                                    requires_grad=True)

        self.vertices_init = vertices_init
        self.faces = faces
        self.uvs = uvs
        self.face_uvs = face_uvs
        self.face_uvs_idx = face_uvs_idx
        self.vertices_laplacian_matrix = vertices_laplacian_matrix

    def set_dataloader(self, images, masks, cameras_info):
        train_data = []
        for cnt, image, mask, camera_info in enumerate(zip(images, masks, cameras_info)):
            data = {}
            data['rgb'] = torch.from_numpy(image)[:, :, :3].float() / 255.
            data['sementic'] = torch.from_numpy(mask)
            data['view_num'] = cnt
            train_data.append(data)

        self.dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                 shuffle=True, pin_memory=True)
        return

    def render(self, cam_transform):

        device = cam_transform.device
        cam_proj = self.cam_proj.to(device)
        faces = self.faces.to(device)

        # object_pos = torch.tensor([[0., 0., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        # camera_up = torch.tensor([[0., 1., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)

        ### Prepare mesh data with projection regarding to camera ###
        vertices_batch = recenter_vertices(self.vertices, self.vertice_shift).to(device)

        face_vertices_camera, face_vertices_image, face_normals = \
            prepare_vertices(vertices=vertices_batch.repeat(self.batch_size, 1, 1),
                             faces=faces, camera_proj=cam_proj, camera_transform=cam_transform
                             )

        face_attributes = [
            self.face_uvs.repeat(self.batch_size, 1, 1, 1),
            torch.ones((self.batch_size, self.num_faces, 3, 1), device='cuda')
        ]

        image_features, soft_mask, face_idx = dibr_rasterization(
            self.image_size, self.image_size, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        image = kal.render.mesh.texture_mapping(texture_coords,
                                                self.textures.repeat(self.batch_size, 1, 1, 1),
                                                mode='bilinear')
        render_img = image.permute(0, 3, 1, 2)
        render_silhouttes = soft_mask

        return render_img, render_silhouttes

    def train(self, camera_info):
        ## Separate vertices center as a learnable parameter
        vertices_init = self.vertices_init
        vertices_init.requires_grad = False

        # This is the center of the optimized mesh, separating it as a learnable parameter helps the optimization.
        self.vertice_shift = torch.zeros((3,), dtype=torch.float, device='cuda',
                                    requires_grad=True)

        camera_pos = camera_info[0].cuda()
        object_pos = camera_info[1].cuda()
        camera_up = camera_info[2].cuda()

        camera_pos.requires_grad = True
        object_pos.requires_grad = True
        camera_up.requires_grad = True

        # Set optimizer and scheduler
        vertices_optim = torch.optim.Adam(params=[self.vertices, self.vertice_shift],
                                          lr=self.vertice_lr)
        texture_optim = torch.optim.Adam(params=[self.textures], lr=self.texture_lr)

        camera_optim = torch.optim.Adam(params=[camera_pos, object_pos, camera_up], lr=self.camera_lr)

        vertices_scheduler = torch.optim.lr_scheduler.StepLR(
            vertices_optim,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma)
        texture_scheduler = torch.optim.lr_scheduler.StepLR(
            texture_optim,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma)
        camera_scheduler = torch.optim.lr_scheduler.StepLR(
            camera_optim,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma)

        ## start train
        for epoch in range(self.num_epoch):
            for idx, data in enumerate(self.dataloader):
                vertices_optim.zero_grad()
                texture_optim.zero_grad()
                camera_optim.zero_grad()

                gt_imgs = data['rgb'].cuda()
                gt_masks = data['semantic'].cuda()

                cam_transform = generate_transformation_matrix(camera_pos[data['view_num']], object_pos[data['view_num']], camera_up[data['view_num']])

                pred_imgs, pred_masks = self.render(self.vertices, self.texture_map, cam_transform)

                ### Compute Losses ###
                # img, mask loss
                loss = self.calc_img_loss(pred_imgs, pred_masks, gt_imgs, gt_masks.squeeze(1))

                # mesh regularization
                loss_lap = self.calc_lap_loss()
                loss_mov = self.calc_mov_loss() / 3.

                total_loss = loss + loss_lap + loss_mov
                ### Update the mesh ###
                total_loss.backward()

                vertices_optim.step()
                texture_optim.step()
                camera_optim.step()

            vertices_scheduler.step()
            texture_scheduler.step()
            camera_scheduler.step()

            print(f"Epoch {epoch} - loss: {float(loss)}")

    def calc_img_loss(self, pred_img, pred_mask, gt_img, gt_mask):
        loss_image = torch.mean(torch.abs(pred_img - gt_img))
        loss_mask = kal.metrics.render.mask_iou(pred_mask, gt_mask)
        loss_perceptual = self.texture_loss(pred_img, gt_img)

        loss_data = self.image_weight * loss_image + self.mask_weight * loss_mask + self.perceptual_wight * loss_perceptual
        return loss_data

    def calc_mov_loss(self):
        Na = self.vertices - self.vertices_init
        Nf = Na.index_select(1, self.flip_index.to(Na.device))
        Nf[..., 2] *= -1

        loss_norm = (Na - Nf).norm(dim=2).mean()
        return self.mov_weight * loss_norm

    def calc_lap_loss(self):
        # laplacian loss
        delta_vertices = self.vertices - self.vertices_init
        device = delta_vertices.device
        nb_vertices = delta_vertices.shape[1]

        vertices_laplacian_matrix = self.vertices_laplacian_matrix.to(device)

        delta_vertices_laplacian = torch.matmul(vertices_laplacian_matrix, delta_vertices)
        loss_laplacian = torch.mean(delta_vertices_laplacian ** 2) * nb_vertices * 3

        loss_reg = self.laplacian_weight * loss_laplacian
        return loss_reg

    def create_3d_object(self, vertices, textures, images, masks, cameras_info, category):
        # path to the rendered image (using the data synthesizer)

        save_path = f'./save/{category}/'

        # dataloader 세팅
        self.set_dataloader(images, masks, cameras_info)

        # vertices, textures 저장
        self.vertices = vertices
        self.textures = textures

        # train
        self.train()

        self.save_object(self.vertices, self.textures, cameras_info, category, save_path)

        return save_path

    def save_object(self, vertices, textures, cameras_pos, category, save_path):
        # path to the rendered image (using the data synthesizer)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # vertices, textures 저장
        self.vertices = vertices
        self.textures = textures

        # save object
        self.export_into_gltf(save_path, category)

        # save thumbnail img
        self.save_thumbnail(save_path, cameras_pos)

        return

    def save_thumbnail(self, save_path, camera_pos):
        # This is similar to a training iteration (without the loss part)
        object_pos = torch.tensor([[0., 0., 0.]], dtype=torch.float).cuda()
        camera_up = torch.tensor([[0., 1., 0.]], dtype=torch.float).cuda()

        cam_transform = kal.render.camera.generate_transformation_matrix(camera_pos, object_pos, camera_up)

        cam_proj = self.cam_proj.cuda()

        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                self.vertices, self.faces, cam_proj, camera_transform=cam_transform
            )

        face_attributes = [
            self.face_uvs.repeat(self.test_batch_size, 1, 1, 1),
            torch.ones((self.test_batch_size, self.num_faces, 3, 1), device='cuda'),
        ]

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            self.image_size, self.image_size, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        texture_coords, mask = image_features
        image = kal.render.mesh.texture_mapping(texture_coords,
                                                self.textures.repeat(self.test_batch_size, 1, 1, 1),
                                                mode='bilinear')

        image = (torch.clamp(image * mask, 0., 1.) + torch.ones_like(image) * (1 - mask)) * 255

        cv2.imwrite(f"{save_path}/thumbnail.png", image[0].cpu().detach().numpy())

        return

    def export_into_gltf(self, save_path, category):
        time = Usd.TimeCode.Default()

        # 저장할 object name setting
        mesh_name = f'mesh_{category}'
        ind_out_path = posixpath.join(save_path, f'{mesh_name}.usdc')

        # save texture
        img_tensor = torch.clamp(self.textures[0], 0., 1.)
        img_tensor_uint8 = (img_tensor * 255.).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        img = Image.fromarray(img_tensor_uint8.squeeze().cpu().numpy())

        texture_dir = posixpath.join(save_path, 'textures')
        rel_filepath = posixpath.join('textures', f'{mesh_name}_diffuse.png')

        if not os.path.exists(texture_dir):
            os.makedirs(texture_dir)

        texture_file = f'{mesh_name}_diffuse.png'
        img.save(posixpath.join(texture_dir, texture_file))

        # create stage
        scene_path = "/TexModel"
        print(ind_out_path)
        if os.path.exists(ind_out_path):
            stage = Usd.Stage.Open(ind_out_path)
        else:
            stage = Usd.Stage.CreateNew(ind_out_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        modelRoot = UsdGeom.Xform.Define(stage, scene_path)
        Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

        # mesh
        usd_mesh = UsdGeom.Mesh.Define(stage, f'{scene_path}/{category}')

        # face
        num_faces = self.faces.size(0)
        face_vertex_counts = [self.faces.size(1)] * num_faces
        faces_list = self.faces.view(-1).cpu().long().numpy()
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces_list, time=time)

        # vertices
        vertices_list = self.vertices.detach().cpu().float().numpy()
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices_list), time=time)

        # uvs
        uvs_list = self.uvs.view(-1, 2).detach().cpu().float().numpy()
        pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            'st', Sdf.ValueTypeNames.TexCoord2fArray)
        pv.Set(uvs_list, time=time)
        pv.SetInterpolation('faceVarying')
        pv.SetIndices(Vt.IntArray.FromNumpy(self.face_uvs_idx.view(-1).cpu().long().numpy()), time=time)

        # material
        material_path = f'{scene_path}/{category}Mat'
        material = UsdShade.Material.Define(stage, material_path)

        pbrShader = UsdShade.Shader.Define(stage, f'{material_path}/PBRShader')
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        stReader = UsdShade.Shader.Define(stage, f'{material_path}/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')

        diffuseTextureSampler = UsdShade.Shader.Define(stage, f'{material_path}/diffuseTexture')
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(f"{rel_filepath}")
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(),
                                                                                           'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            diffuseTextureSampler.ConnectableAPI(), 'rgb')

        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
        UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)

        stage.Save()

        save_file_path = f"{save_path}/{category}.gltf"
        # bin_path = f"{save_path}buffer.bin"

        scn = a3d.Scene()
        scn.open(ind_out_path)
        scn.save(save_file_path, a3d.FileFormat.GLTF2)

        return