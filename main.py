import sys
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import torch

sys.path.append('./mask')
sys.path.append('./style_gan')
sys.path.append('./predictor')

from style_gan import style_gan
from mask import mask
from predictor import predictor

# start : uvicorn main:app --reload
app = FastAPI()

categories = ['ring','shirts','pants','hat','necklace','bag']

# style_gan 모델 불러오기
style_gan_network_path = "./style_gan/network/shirts.pkl"
style_gan_model = style_gan.get_models(style_gan_network_path)

# 3d 속성 predictor 모델, diffRender 불러오기
predictor_model_path = './predictor/network/latest_ckpt.pth'

# TODO: 카테고리 별로 init_mesh 다르게 설정하기
init_mesh_path = './predictor/samples/sphere.obj'
predictor_model, diffRender = predictor.get_predictor_model(init_mesh_path,predictor_model_path,style_gan_model.img_resolution)

# mask_rcnn 모델 불러오기
mask_weights_path = "./mask/weight/mask_rcnn_matilda_0110.h5"
mask_rcnn_model = mask.get_mask_model(mask_weights_path)

# 카메라 정보 불러오기
cameras_info = load_cameras_info('./predictor/samples/')

def save_file_into_store(path):
    return

def load_cameras_info(root):
    cameras_info = {c for c in categories}
    for category in categories:
        cameras_info[category] = np.load(f'{root}{category}.npy')
    return cameras_info

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

    # image를 넣어 mesh, texture 생성
    attributes = predictor_model(torch.Tensor(image.transpose(2, 0, 1)).unsqueeze(0))

    mesh = attributes['vertices']
    texture = attributes['textures']
    #lights = attributes['lights']

    # mesh, texture를 style gan network에 넣어 다각도 이미지 생성
    mv_images = style_gan.get_multiview_images(style_gan_model, torch.Tensor(cameras_info[category]), mesh, texture)

    # TODO: mask_rcnn을 사용할 지, IS_NET을 사용할 지 결정
    ''' in mask.py '''
    # 이미지들의 sementic mask 얻기
    mv_masks = mask.detect_mask(mask_rcnn_model,[mv_images])

    ''' in dibr.py '''
    # 3D Object 생성
    bin_path, obj_path, thumb_nail_img = diffRender.create_3d_object(mesh, texture, mv_images, mv_masks, cameras_info[category], category)
    
    # 3D Object를 저장소에 저장
    save_file_into_store(bin_path)
    save_file_into_store(obj_path)

    return {"file_name": file.filename, "category" : category}
