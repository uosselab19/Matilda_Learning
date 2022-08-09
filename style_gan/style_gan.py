import copy
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

import dnnlib
import legacy

device = torch.device('cuda')

def get_multiview_images(G, cameras, mesh, texture):
    """Generate images using pretrained network."""
    image_list = []
    print('Generating style-mixed images...')
    for camera in cameras:
        image = G.synthesis(camera, mesh, texture)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_list.append(image[0].cpu().numpy())

    return image_list

def get_models(network_pkl):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G
