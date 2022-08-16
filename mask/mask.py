import torch
import torch.nn.functional as F

from mask.DIS.models import isnet

def get_mask_model(model_path):
    print(f"restore model from: {model_path}")
    net = isnet.ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
    else:
        print('can not load pretrained_network')
        exit(0)
    net.eval()
    return net

def get_mask_from_image(net, image):
    # image should be (1, 3, 512, 512)
    print("Making Mask...")
    ds_val = net(image)[0]

    pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        F.upsample(torch.unsqueeze(pred_val, 0), (512, 512), mode='bilinear'))

    # pred_val = normPRED(pred_val)
    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    return pred_val