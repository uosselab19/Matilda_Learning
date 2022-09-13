import torch
import torch.nn.functional as F

from mask.DIS.models import isnet
from torchvision.transforms.functional import normalize
from torchvision.transforms.functional import to_pil_image
import torchvision

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
    im_shp=im_tensor.shape[0:2]
    im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

    if torch.cuda.is_available():
        image=image.cuda()

    result=net(image)
    result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = to_pil_image((result-mi)/(ma-mi))
    result = result.point(lambda p: p >= 60 and 255)  # 하얀색으로
    result = torchvision.transforms.functional.to_tensor(result).max(0, True)[0]

    return result