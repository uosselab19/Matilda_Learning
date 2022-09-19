import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def net_init(modules):
    for m in modules:
        if isinstance(m, nn.ConvTranspose2d) \
        or isinstance(m, nn.Linear) \
        or isinstance(object, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0, std=0.001)

def camera_position_from_spherical_angles(dist, elev, azim, degrees=True):
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.
    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.
    The vectors are broadcast against each other so they all have shape (N, 1).
    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    return camera_position.reshape(-1, 3)

def convblock(indim, outdim, ker, stride, pad):
    block2 = [
        nn.Conv2d(indim, outdim, ker, stride, pad),
        nn.BatchNorm2d(outdim),
        nn.ReLU()
    ]
    return block2


def linearblock(indim, outdim):
    block2 = [
        nn.Linear(indim, outdim),
        nn.BatchNorm1d(outdim),
        nn.ReLU()
    ]
    return block2


class AttBlock(nn.Module):
    def __init__(self, nc, nk, nf, out_dim, method):
        super(AttBlock, self).__init__()
        self.method = method

        block1 = convblock(nc, nf // 2, nk, stride=2, pad=2)
        block2 = convblock(nf // 2, nf, nk, stride=2, pad=2)
        block3 = convblock(nf, nf * 2, nk, stride=2, pad=2)
        block4 = convblock(nf * 2, nf * 4, nk, stride=2, pad=2)
        block5 = convblock(nf * 4, nf * 8, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        if method == 'linear':
            linear1 = linearblock(nf * 8, nf * 16)
            linear2 = linearblock(nf * 16, nf * 16)
            self.linear3 = nn.Linear(nf * 16, out_dim)
        elif method == 'conv':
            linear1 = convblock(nf * 8, nf * 16, 1, stride=1, pad=0)
            linear2 = convblock(nf * 16, nf * 8, 1, stride=1, pad=0)
        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        net_init(self.modules())

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, \
            linear1, linear2

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.encoder1(x)

        if self.method == 'linear':
            x = x.view(batch_size, -1)
            x = self.encoder2(x)
            x = self.linear3(x)
        else:
            x = x.view(batch_size, -1, 1, 1)
            x = self.encoder2(x)
        return x


class ShapeEncoder(nn.Module):
    def __init__(self, nc, nk, nf, num_vertices):
        super(ShapeEncoder, self).__init__()
        self.num_vertices = num_vertices

        self.enc = AttBlock(nc=nc, nk=nk, nf=nf, out_dim=self.num_vertices * 3, method='linear')

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.enc(x)

        delta_vertices = x.view(batch_size, self.num_vertices, 3)
        delta_vertices = torch.tanh(delta_vertices)
        return delta_vertices


class LightEncoder(nn.Module):
    def __init__(self, nc, nf, nk):
        super(LightEncoder, self).__init__()

        self.enc = AttBlock(nc=nc, nk=nk, nf=nf, out_dim=9, method='linear')

    def forward(self, x):
        x = self.enc(x)

        lightparam = torch.tanh(x)
        scale = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32).cuda()
        bias = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
        lightparam = lightparam * scale + bias

        return lightparam


class TextureEncoder(nn.Module):
    def __init__(self, nc, nf, nk, num_vertices):
        super(TextureEncoder, self).__init__()
        self.num_vertices = num_vertices

        self.enc = AttBlock(nc=nc, nk=nk, nf=nf, out_dim=None, method='conv')

        self.texture_flow = nn.Sequential(
            # input is Z, going into a convolution
            nn.Upsample(scale_factor=4),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),

            # state size. (nf*8) x 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            # state size. (nf*4) x 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            # state size. (nf*2) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (nf) x 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (nf) x 128 x 128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 2, nf, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            # state size. (nf) x 256 x 256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf, 2, 3, 1, 1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x, flip_dim):
        img = x[:, :3]
        x = self.enc(x)
        uv_sampler = self.texture_flow(x).permute(0, 2, 3, 1)
        textures = F.grid_sample(img, uv_sampler)

        textures_flip = textures.flip([flip_dim])
        textures = torch.cat([textures, textures_flip], dim=flip_dim)
        return textures


class CameraEncoder(nn.Module):
    def __init__(self, nc, nk, nf, azi_scope, elev_range, dist_range):
        super(CameraEncoder, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        self.elev_min = float(elev_range[0])
        self.elev_max = float(elev_range[1])

        dist_range = dist_range.split('~')
        self.dist_min = float(dist_range[0])
        self.dist_max = float(dist_range[1])

        self.pred_cam_pos_encoder = AttBlock(nc=nc, nk=nk, nf=nf, out_dim=4, method='linear')

    def atan2(self, y, x):
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-12) + 1e-6
        phi = torch.sign(y) * torch.acos(x / r) * 180.0 / math.pi
        return phi

    def forward(self, feat):
        camera_pos_att = self.pred_cam_pos_encoder.forward(feat)

        # cameras
        distances = self.dist_min + torch.sigmoid(camera_pos_att[:, 0]) * (self.dist_max - self.dist_min)
        elevations = self.elev_min + torch.sigmoid(camera_pos_att[:, 1]) * (self.elev_max - self.elev_min)

        azimuths_x = camera_pos_att[:, 2]
        azimuths_y = camera_pos_att[:, 3]
        azimuths = - self.atan2(azimuths_y, azimuths_x) / 360.0 * self.azi_scope

        cameras = [distances, elevations, azimuths]

        return cameras

class AttributeEncoder(nn.Module):
    def __init__(self, num_vertices, vertices_init, nc, nf, nk, azi_scope, elev_range, dist_range, scale=2.):
        super(AttributeEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init
        self.scale = scale

        self.shape_enc = ShapeEncoder(nc=nc, nk=nk, nf=nf * 2, num_vertices=self.num_vertices)
        self.texture_enc = TextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices)
        self.light_enc = LightEncoder(nc=nc, nk=nk, nf=nf * 2)
        self.camera_enc = CameraEncoder(nc=nc, nk=nk, nf=nf, azi_scope=azi_scope, elev_range=elev_range,
                                        dist_range=dist_range)

    def forward(self, x, flip_dim=2):
        device = x.device
        input_img = x

        # cameras
        distances, elevations, azimuths = self.camera_enc(input_img)

        # vertex
        delta_vertices = self.shape_enc(input_img) * self.scale
        vertices = self.vertices_init[None].to(device) + delta_vertices

        # textures
        textures = self.texture_enc(input_img,flip_dim)
        lights = self.light_enc(input_img)

        # others
        attributes = {
            'distances': distances,
            'elevations': elevations,
            'azimuths': azimuths,
            'vertices': vertices,
            'delta_vertices': delta_vertices,
            'textures': textures,
            'lights': lights
        }
        return attributes