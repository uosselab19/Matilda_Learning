# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

import dnnlib
import legacy

device = torch.device('cuda')

def get_multiview_images(
    G,
    w_origin,
    w_views,
    view_styles : List[int],
    noise_mode: str,
):
    """Generate images using pretrained network."""

    image_list = []
    print('Generating style-mixed images...')
    for w_view in w_views:
        w = w_origin.clone()
        w[view_styles] = w_view[view_styles]
        image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_list.append(image[0].cpu().numpy())

    return image_list

def get_models(network_pkl):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    return G, vgg16
