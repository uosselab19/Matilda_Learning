from predictor.network import networks
# import kaolin related
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection, generate_transformation_matrix
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices

# import
import os
import torch
import time
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import trimesh

import aspose.threed as a3d
from PIL import Image
import numpy as np

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt

### get_model ###
def get_predictor_model(template_path, resume_path, args):
    assert os.path.exists(resume_path)
    image_size = args['image_size']
    diffRender = DiffRender(filename_obj=template_path, image_size=image_size)

    # netE: 3D attribute encoder: Light, Shape, and Texture
    azi_scope = args['azi_scope']
    elev_range = args['elev_range']
    dist_range = args['dist_range']
    scale = args['scale']
    netE = networks.AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, \
                                     nc=3, nk=5, nf=32, azi_scope=azi_scope, elev_range=elev_range, dist_range=dist_range, scale=scale)
    netE = netE.cuda()

    print("=> loading checkpoint '{}'".format(resume_path))
    # Map model to be loaded to specified single gpu.
    checkpoint = torch.load(resume_path)
    netE.load_state_dict(checkpoint['netE'])
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    netE.eval()

    return netE, diffRender

#####################################################################
# ----------------------------- DIB_R ----------------------------- #
#####################################################################

class DiffRender(object):
    batch_size = 1

    def __init__(self, filename_obj, image_size):
        self.image_size = image_size
        # camera projection matrix
        camera_fovy = np.arctan(1.0 / 2.5) * 2
        self.cam_proj = generate_perspective_projection(camera_fovy, ratio=image_size / image_size)

        mesh = kal.io.obj.import_mesh(filename_obj, with_materials=True)

        # get vertices_init
        vertices = mesh.vertices
        vertices.requires_grad = False
        vertices_init = vertices.clone().detach() # (V, 3)

        # get face_uvs
        faces = mesh.faces.cuda()
        uvs = mesh.uvs.cuda().unsqueeze(0)
        face_uvs_idx = mesh.face_uvs_idx.cuda()
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
        face_uvs.requires_grad = False

        self.num_faces = faces.shape[0]
        self.num_vertices = vertices_init.shape[0]

        self.vertices_init = vertices_init
        self.faces = faces
        self.uvs = uvs
        self.face_uvs = face_uvs
        self.face_uvs_idx = face_uvs_idx

    def render(self, camera_pos):

        device = camera_pos.device
        cam_proj = self.cam_proj.to(device)
        faces = self.faces.to(device)

        object_pos = torch.tensor([[0., 0., 0.]], dtype=torch.float).cuda()
        camera_up = torch.tensor([[0., 1., 0.]], dtype=torch.float).cuda()

        cam_transform = kal.render.camera.generate_transformation_matrix(camera_pos, object_pos, camera_up)

        face_vertices_camera, face_vertices_image, face_normals = \
            prepare_vertices(vertices=self.vertices,
                             faces=faces, camera_proj=cam_proj, camera_transform=cam_transform
                             )

        face_normals_unit = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
        face_normals_unit = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)
        face_attributes = [
            torch.ones((self.batch_size, self.num_faces, 3, 1), device=device),
            self.face_uvs.repeat(self.batch_size, 1, 1, 1),
            face_normals_unit
        ]

        image_features, soft_mask, face_idx = dibr_rasterization(
            self.image_size, self.image_size, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texmask, texcoord, imnormal = image_features

        texcolor = texture_mapping(texcoord, self.textures, mode='bilinear')
        coef = spherical_harmonic_lighting(imnormal, self.lights)
        image = texcolor * texmask * coef.unsqueeze(-1) + torch.ones_like(texcolor) * (1 - texmask)
        image = torch.clamp(image, 0, 1)
        render_img = image.permute(0, 3, 1, 2)
        render_silhouttes = soft_mask

        return render_img, render_silhouttes

    def save_object(self, attributes, category, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # vertices, textures, lights 저장
        self.vertices = attributes['vertices']
        self.textures = attributes['textures']
        self.lights = attributes['lights']

        # save object
        obj_save_path = self.export_into_glb(save_path, category)

        # tri_mesh = trimesh.Trimesh(self.vertices[0].detach().cpu ().numpy(), self.faces.detach().cpu().numpy())
        # texure_maps = to_pil_image(self.textures[0].detach().cpu())
        # texure_maps.save('./test_texture.png', 'PNG')
        # tri_mesh.export('./test.obj')

        # camera propertis : distances, elevations, azimuths
        distances = attributes['distances']
        elevations = attributes['elevations'] + 5.0
        azimuths = attributes['azimuths'] + 20.0
        cameras_pos = networks.camera_position_from_spherical_angles(distances,elevations,azimuths)
        
        # save thumbnail img
        img_save_path = self.save_thumbnail(save_path, cameras_pos)

        return obj_save_path, img_save_path

    def save_thumbnail(self, save_path, camera_pos):
        img_save_path = save_path + 'img.png'

        image, mask = self.render(camera_pos)

        image = (torch.clamp(image * mask, 0., 1.) + torch.ones_like(image) * (1 - mask)) * 255

        test_mask = to_pil_image(mask.detach().cpu())
        test_mask.save('./test_mask.png', 'PNG')

        vutils.save_image(image.detach(), img_save_path, normalize=True)

        return img_save_path

    def export_into_glb(self, save_path, category):
        time = Usd.TimeCode.Default()

        # object usdc file path
        usd_path = save_path + 'mesh.usdc'

        # save texture
        img_tensor = torch.clamp(self.textures[0], 0., 1.)
        img_tensor_uint8 = (img_tensor * 255.).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        img = Image.fromarray(img_tensor_uint8.squeeze().cpu().numpy())

        rel_filepath = save_path + 'diffuse.png'
        img.save(rel_filepath)

        # create stage
        scene_path = "/TexModel"
        if os.path.exists(usd_path):
            stage = Usd.Stage.Open(usd_path)
        else:
            stage = Usd.Stage.CreateNew(usd_path)
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

        scn = a3d.Scene()
        scn.open(usd_path)
        obj_save_path = save_path + 'obj.glb'
        scn.save(obj_save_path, a3d.FileFormat.GLTF2_BINARY)

        return obj_save_path