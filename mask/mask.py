import torch
import torch.nn.functional as F

from mask.DIS.models import isnet
from torchvision.transforms.functional import normalize
from torchvision.transforms.functional import to_pil_image
import torchvision
import numpy as np
from skimage import io

def get_mask_model(model_path):
    print(f"restore model from: {model_path}")
    net = isnet.ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
    else:
        print('can not load pretrained_network')
        exit(0)
    net = net.cuda()
    net.eval()
    return net

def get_mask_from_image(net, im_tensor):
    print("Making Mask...")
    input_size = [1024, 1024]
    im_shp=im_tensor.shape[1:3]
    # image = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear")

    img_mask = net(im_tensor.unsqueeze(0))[0]
    img_mask = img_mask[0][0,:,:,:]
    img_mask = torch.squeeze(
        F.upsample(torch.unsqueeze(img_mask, 0),im_shp, mode='bilinear'))

    ma = torch.max(img_mask)
    mi = torch.min(img_mask)
    img_mask = (img_mask-mi)/(ma-mi)
    img_mask = to_pil_image(img_mask)
    img_mask = img_mask.point(lambda p: p >= 60 and 255)  # 하얀색으로
    img_mask = torchvision.transforms.functional.to_tensor(img_mask).max(0, True)[0].cuda()

    return img_mask