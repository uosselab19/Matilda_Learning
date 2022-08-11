# import torch related
import torch
import torch.nn as nn
#import torch.nn.functional as F

def deep_copy(att, index=None, detach=False):
    if index is None:
        index = torch.arange(att['textures'].shape[0]).cuda()

    copy_att = {}
    for key, value in att.items():
        copy_keys = ['vertices', 'delta_vertices', 'textures', 'lights']
        if key in copy_keys:
            if detach:
                copy_att[key] = value[index].clone().detach()
            else:
                copy_att[key] = value[index].clone()
    return copy_att


def convblock(indim, outdim, ker, stride, pad):
    block = [
        nn.Conv2d(indim, outdim, ker, stride, pad),
        nn.BatchNorm2d(outdim),
        nn.ReLU()
    ]
    return block


def convtransblock(indim, outdim, ker, stride, pad, scale_factor=2, padding_mode='zeros'):
    block = [
        nn.Upsample(scale_factor=scale_factor),
        nn.Conv2d(indim, outdim, ker, stride, pad, padding_mode=padding_mode),
        nn.BatchNorm2d(outdim),
        nn.ReLU(True),
    ]
    return block


def linearblock(indim, outdim):
    block = [
        nn.Linear(indim, outdim),
        nn.BatchNorm1d(outdim),
        nn.ReLU()
    ]
    return block


class ShapeEncoder(nn.Module):
    def __init__(self, nc, nk, num_vertices):
        super(ShapeEncoder, self).__init__()
        self.num_vertices = num_vertices

        block1 = convblock(nc, 32, nk, stride=2, pad=2)
        block2 = convblock(32, 64, nk, stride=2, pad=2)
        block3 = convblock(64, 128, nk, stride=2, pad=2)
        block4 = convblock(128, 256, nk, stride=2, pad=2)
        block5 = convblock(256, 512, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = linearblock(512, 1024)
        linear2 = linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, self.num_vertices * 3)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1, linear2

    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)

        x = self.linear3(x)

        delta_vertices = x.view(batch_size, self.num_vertices, 3)
        delta_vertices = torch.tanh(delta_vertices)
        return delta_vertices


class LightEncoder(nn.Module):
    def __init__(self, nc, nk):
        super(LightEncoder, self).__init__()

        block1 = convblock(nc, 32, nk, stride=2, pad=2)
        block2 = convblock(32, 64, nk, stride=2, pad=2)
        block3 = convblock(64, 128, nk, stride=2, pad=2)
        block4 = convblock(128, 256, nk, stride=2, pad=2)
        block5 = convblock(256, 512, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = linearblock(512, 1024)
        linear2 = linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 9)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1, linear2

    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)
        x = self.linear3(x)

        lightparam = torch.tanh(x)
        scale = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32).cuda()
        bias = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
        lightparam = lightparam * scale + bias

        return lightparam


class TextureEncoder(nn.Module):
    def __init__(self, nc, nf, nk, num_vertices):
        super(TextureEncoder, self).__init__()
        self.num_vertices = num_vertices

        block1 = convblock(nc, nf // 2, nk, stride=2, pad=2)
        block2 = convblock(nf // 2, nf, nk, stride=2, pad=2)
        block3 = convblock(nf, nf * 2, nk, stride=2, pad=2)
        block4 = convblock(nf * 2, nf * 4, nk, stride=2, pad=2)
        block5 = convblock(nf * 4, nf * 8, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = convblock(nf * 8, nf * 16, 1, stride=1, pad=0)
        linear2 = convblock(nf * 16, nf * 8, 1, stride=1, pad=0)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        tex_block1 = convtransblock(nf * 8, nf * 8, 3, stride=1, pad=1, scale_factor=4)
        tex_block2 = convtransblock(nf * 8, nf * 8, 3, stride=1, pad=1, scale_factor=2, padding_mode='reflect')
        tex_block3 = convtransblock(nf * 8, nf * 4, 3, stride=1, pad=1, scale_factor=2, padding_mode='reflect')
        tex_block4 = convtransblock(nf * 4, nf * 2, 3, stride=1, pad=1, scale_factor=2, padding_mode='reflect')
        tex_block5 = convtransblock(nf * 2, nf * 2, 3, stride=1, pad=1, scale_factor=2, padding_mode='reflect')
        tex_block6 = convtransblock(nf * 2, nf, 3, stride=1, pad=1, scale_factor=2, padding_mode='reflect')
        tex_block_last = [nn.Upsample(scale_factor=2),
                          nn.Conv2d(nf, 3, 3, 1, 1, padding_mode='reflect'),
                          nn.Tanh()]

        all_blocks = tex_block1 + tex_block2 + tex_block3 + tex_block4 + tex_block5 + tex_block6 + tex_block_last
        self.texture_flow = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, linear1, linear2, tex_block1, tex_block2, tex_block3, tex_block4, tex_block5, tex_block_last

    def forward(self, x):
        img = x[:, :3]
        batch_size = x.shape[0]
        x = self.encoder1(x)
        x = x.view(batch_size, -1, 1, 1)
        x = self.encoder2(x)
        textures = (self.texture_flow(x) + 1) / 2  # (batch_size, 3, 256, 256)

        return textures


class AttributeEncoder(nn.Module):
    def __init__(self, num_vertices, vertices_init, nc, nf, nk):
        super(AttributeEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init

        self.shape_enc = ShapeEncoder(nc=nc, nk=nk, num_vertices=self.num_vertices)
        self.texture_enc = TextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices)
        self.light_enc = LightEncoder(nc=nc, nk=nk)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        input_img = x

        # vertex
        delta_vertices = self.shape_enc(input_img)
        vertices = self.vertices_init[None].to(device) + delta_vertices

        # textures
        textures = self.texture_enc(input_img)
        lights = self.light_enc(input_img)

        # others
        attributes = {
            'vertices': vertices,
            'delta_vertices': delta_vertices,
            'textures': textures,
            'lights': lights
        }
        return attributes