import os
from glob import glob
import time

import skimage
from skimage import io
import aspose.threed as a3d
from PIL import Image
import torch
import numpy as np

import kaolin as kal
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt
import posixpath

# Hyperparameters
num_epoch = 100
batch_size = 1
laplacian_weight = 0.05
flat_weight = 0.001
image_weight = 0.1
mask_weight = 1.0
lr = 3e-2
scheduler_step_size = 15
scheduler_gamma = 0.5

texture_res = 400

# select camera angle for thumbnail
thumb_nail_id = [0]
thumb_nail_size = len(thumb_nail_id)

class Mesh():
    def __init__(self, vertices, faces, uvs, face_uvs, face_uvs_idx):
        self.vertices = vertices,
        self.faces = faces,
        self.uvs = uvs,
        self.face_uvs = face_uvs,
        self.face_uvs_idx = face_uvs_idx,

def set_dataloader(images, masks, cameras_info):
    train_data = []
    for image, mask, camera_info in zip(images, masks, cameras_info):
        data = {}
        data['rgb'] = torch.from_numpy(image)[:, :, :3].float() / 255.
        data['sementic'] = torch.from_numpy(mask)
        data['metadata'] = camera_info
        train_data.append(data)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                             shuffle=True, pin_memory=True)
    return dataloader

def get_mesh_info(path, scale):
    mesh = kal.io.obj.import_mesh(path, with_materials=True)
    # the sphere is usually too small (this is fine-tuned for the clock)
    vertices = mesh.vertices.cuda().unsqueeze(0) * scale
    vertices.requires_grad = True
    faces = mesh.faces.cuda()
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()

    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
    face_uvs.requires_grad = False

    return Mesh(vertices,faces,uvs,face_uvs,face_uvs_idx)

def get_edges_to_face(mesh, face_size, nb_faces):
    ## Set up auxiliary connectivity matrix of edges to faces indexes for the flat loss
    edges = torch.cat([mesh.faces[:, i:i + 2] for i in range(face_size - 1)] +
                      [mesh.faces[:, [-1, 0]]], dim=0)

    edges = torch.sort(edges, dim=1)[0]
    face_ids = torch.arange(nb_faces, device='cuda', dtype=torch.long).repeat(face_size)
    edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
    nb_edges = edges.shape[0]
    # edge to faces
    sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
    sorted_faces_ids = face_ids[order_edges_ids]
    # indices of first occurences of each key
    idx_first = torch.where(
        torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                                (1, 0), value=1))[0]
    nb_faces_per_edge = idx_first[1:] - idx_first[:-1]
    # compute sub_idx (2nd axis indices to store the faces)
    offsets = torch.zeros(sorted_edges_ids.shape[0], device='cuda', dtype=torch.long)
    offsets[idx_first[1:]] = nb_faces_per_edge
    sub_idx = (torch.arange(sorted_edges_ids.shape[0], device='cuda', dtype=torch.long) -
               torch.cumsum(offsets, dim=0))
    nb_faces_per_edge = torch.cat([nb_faces_per_edge,
                                   sorted_edges_ids.shape[0] - idx_first[-1:]],
                                  dim=0)
    max_sub_idx = 2
    edge2faces = torch.zeros((nb_edges, max_sub_idx), device='cuda', dtype=torch.long)
    edge2faces[sorted_edges_ids, sub_idx] = sorted_faces_ids

    return edge2faces

def recenter_vertices(vertices, vertice_shift):
    """Recenter vertices on vertice_shift for better optimization"""
    vertices_min = vertices.min(dim=1, keepdim=True)[0]
    vertices_max = vertices.max(dim=1, keepdim=True)[0]
    vertices_mid = (vertices_min + vertices_max) / 2
    vertices = vertices - vertices_mid + vertice_shift
    return vertices

def train(dataloader,
          mesh
):
    ## Separate vertices center as a learnable parameter
    vertices_init = mesh.vertices.detach()
    vertices_init.requires_grad = False

    # This is the center of the optimized mesh, separating it as a learnable parameter helps the optimization.
    vertice_shift = torch.zeros((3,), dtype=torch.float, device='cuda',
                                requires_grad=True)

    nb_faces = mesh.faces.shape[0]
    nb_vertices = mesh.vertices.shape[1]
    face_size = 3

    edge2faces = get_edges_to_face(mesh,face_size,nb_faces)

    # Set up auxiliary laplacian matrix for the laplacian loss
    vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(
        nb_vertices, mesh.faces)

    # Set optimizer and scheduler
    optim = torch.optim.Adam(params=[mesh.vertices, mesh.texture_map, vertice_shift],
                             lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size,
                                                gamma=scheduler_gamma)

    ## start train
    for epoch in range(num_epoch):
        for idx, data in enumerate(dataloader):
            optim.zero_grad()
            gt_image = data['rgb'].cuda()
            gt_mask = data['semantic'].cuda()
            cam_transform = data['metadata']['cam_transform'].cuda()
            cam_proj = data['metadata']['cam_proj'].cuda()

            ### Prepare mesh data with projection regarding to camera ###
            vertices_batch = recenter_vertices(mesh.vertices, vertice_shift)

            face_vertices_camera, face_vertices_image, face_normals = \
                kal.render.mesh.prepare_vertices(
                    vertices_batch.repeat(batch_size, 1, 1),
                    mesh.faces, cam_proj, camera_transform=cam_transform
                )

            ### Perform Rasterization ###
            # Construct attributes that DIB-R rasterizer will interpolate.
            # the first is the UVS associated to each face
            # the second will make a hard segmentation mask
            face_attributes = [
                mesh.face_uvs.repeat(batch_size, 1, 1, 1),
                torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
            ]

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                gt_image.shape[1], gt_image.shape[2], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])

            # image_features is a tuple in composed of the interpolated attributes of face_attributes
            texture_coords, mask = image_features
            image = kal.render.mesh.texture_mapping(texture_coords,
                                                    mesh.texture_map.repeat(batch_size, 1, 1, 1),
                                                    mode='bilinear')

            ### Compute Losses ###
            image_loss = torch.mean(torch.abs(image - gt_image))
            mask_loss = kal.metrics.render.mask_iou(soft_mask, gt_mask.squeeze(-1))

            # laplacian loss
            vertices_mov = mesh.vertices - vertices_init
            vertices_mov_laplacian = torch.matmul(vertices_laplacian_matrix, vertices_mov)
            laplacian_loss = torch.mean(vertices_mov_laplacian ** 2) * nb_vertices * 3
            # flat loss
            mesh_normals_e1 = face_normals[:, edge2faces[:, 0]]
            mesh_normals_e2 = face_normals[:, edge2faces[:, 1]]
            faces_cos = torch.sum(mesh_normals_e1 * mesh_normals_e2, dim=2)
            flat_loss = torch.mean((faces_cos - 1) ** 2) * edge2faces.shape[0]

            loss = (
                    image_loss * image_weight +
                    mask_loss * mask_weight +
                    laplacian_loss * laplacian_weight +
                    flat_loss * flat_weight
            )
            ### Update the mesh ###
            loss.backward()
            optim.step()

        scheduler.step()
        print(f"Epoch {epoch} - loss: {float(loss)}")

def get_thumbnail():
    # TODO : 미리 지정한 카메라 각도에 해당하는 thumbnail image 생성
    return

def export_into_gltf(save_path,
                mesh,
                category,
):
    time = Usd.TimeCode.Default()

    # 저장할 object name setting
    mesh_name = 'obj'
    ind_out_path = posixpath.join(save_path, f'{mesh_name}.usdc')

    # save texture
    usd_dir = os.path.dirname(ind_out_path)

    texture_dir = 'textures'
    rel_filepath = posixpath.join(texture_dir, f'{mesh_name}_diffuse.png')

    img_tensor = torch.clamp(mesh.texture_map[0], 0., 1.)
    img_tensor_uint8 = (img_tensor * 255.).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
    img = Image.fromarray(img_tensor_uint8.squeeze().cpu().numpy())
    img.save(posixpath.join(usd_dir, rel_filepath))

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
    num_faces = mesh.faces.size(0)
    face_vertex_counts = [mesh.faces.size(1)] * num_faces
    faces_list = mesh.faces.view(-1).cpu().long().numpy()
    usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
    usd_mesh.GetFaceVertexIndicesAttr().Set(faces_list, time=time)

    # vertices
    vertices_list = mesh.vertices.detach().cpu().float().numpy()
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices_list), time=time)

    # uvs
    uvs_list = mesh.uvs.view(-1, 2).detach().cpu().float().numpy()
    pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray)
    pv.Set(uvs_list, time=time)
    pv.SetInterpolation('faceVarying')
    pv.SetIndices(Vt.IntArray.FromNumpy(mesh.face_uvs_idx.view(-1).cpu().long().numpy()), time=time)

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

    save_file_path = f"{save_path}obj_{time}.gltf"
    bin_path = f"{save_path}buffer.bin"

    scn = a3d.Scene()
    scn.open(ind_out_path)
    scn.save(save_file_path, a3d.FileFormat.GLTF2)

    return bin_path, save_file_path

def create_3d_object(images, masks, cameras_info, category):
    # path to the rendered image (using the data synthesizer)

    save_path = f'./save/{category}/'
    mesh_path = './samples/torus.obj'
    mesh_scale = 3

    # dataloader 세팅
    dataloader = set_dataloader(images, masks, cameras_info)

    # 초기 mesh 정보 가져오기
    mesh = get_mesh_info(mesh_path,mesh_scale)

    # texture map 세팅
    mesh.texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                             requires_grad=True)
    # train
    train(dataloader, mesh)

    # save object
    bin_path, obj_file_path = export_into_gltf(save_path, mesh, category)

    # get thumbnail img
    thumb_img = get_thumbnail()

    return bin_path, obj_file_path, thumb_img