import sys
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import torch

sys.path.append('./mask')
sys.path.append('./style_gan')
sys.path.append('./dibr')

from style_gan import style_gan
from mask import mask
from cmr import cmr
from dibr import dibr

# start : uvicorn main:app --reload
app = FastAPI()

categories = ['ring','shirts','pants','hat','necklace','']

# style_gan 모델 불러오기
style_gan_network_path = "style_gan/network/shirts.pkl"
style_gan_model, vgg16 = style_gan.get_models(style_gan_network_path)

# 카메라 정보 불러오기
# cameras_info = load_cameras_info('./dibr/samples/')

# multiview sample w 불러오기
# w_views = load_multiview_ws('./style_gan/samples/')

# mask_rcnn 모델 불러오기
mask_weights_path = "./mask/weight/mask_rcnn_matilda_0110.h5"
mask_rcnn_model = mask.get_mask_model(mask_weights_path)

def save_file_into_store(path):
    return

def load_cameras_info(root):
    cameras_info = {c for c in categories}
    for category in categories:
        cameras_info[category] = np.load(f'{root}{category}.npy')
    return cameras_info

def load_multiview_ws(root):
    ws = {c for c in categories}
    for category in categories:
        ws[category] = np.load(f'{root}{category}.npy')
    return ws

def load_into_numpy_array_and_resize(data, resolution):
    image = Image.open(BytesIO(data))
    w, h = image.size
    s = min(w, h)
    image = image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    image = image.resize((resolution, resolution), Image.LANCZOS)
    image = np.array(image, dtype=np.uint8)
    return image

@app.get("/")
async def root():
    return {"Welcome"}

@app.post("/convert/")
async def convert(file: UploadFile = File(...), category : str = Form(...)):
    ''' in style_gan.py '''
    image = load_into_numpy_array_and_resize(await file.read(),style_gan_model.img_resolution)

    # image를 넣어 camera, mesh, texture 생성
    camera, mesh, texture = cmr.get_object_from_image(style_gan_model,vgg16,torch.Tensor(image.transpose(2, 0, 1)),verbose=True)

    # mesh, texture를 style gan network에 넣어 다각도 이미지 생성
    # col_styles = [0,1,2,3]
    # mv_images = get_multiview_images(style_gan_model, w, w_views[category], col_styles)

    ''' in mask.py '''
    # 이미지들의 sementic mask 얻기
    mv_masks = mask.detect_mask(mask_rcnn_model,[image])

    ''' in dibr.py '''
    # 3D Object 생성
    # bin_path, obj_path, thumb_nail_img = dibr.create_3d_object(mesh, texture, mv_masks, cameras_info[category],category)
    
    # 3D Object를 저장소에 저장
    # save_file_into_store(bin_path)
    # save_file_into_store(obj_path)

    return {"file_name": file.filename, "category" : category}
