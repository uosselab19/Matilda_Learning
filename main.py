import sys
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import numpy as np
import cv2
import torch

sys.path.append('./mask')
sys.path.append('./style_gan')
sys.path.append('./predictor')
sys.path.append('./predictor/PerceptualSimilarity')
sys.path.append('./predictor/network')

from style_gan import style_gan
#from mask import mask
from predictor import predictor

# start : uvicorn main:app --reload --host 0.0.0.0
app = FastAPI()

# categories = ['ring','shirts','pants','hat','necklace','bag'] # TODO: 카테고리 추가
categories = ['shirts'] # for test
#samples_per_categories = {'ring' : 'torus', 'shirts': 'sphere', 'pants': 'sphere', 'hat': 'sphere', 'necklace': 'torus', 'bag': 'torus'}
samples_per_categories = {'shirts' : 'sphere'}

def get_all_models():
    image_size = 512

    style_gan_models = {}
    predictor_models = {}
    diffRenderers = {}

    for category in categories:
        # style_gan
        #style_gan_network_path = f"./style_gan/network/{category}.pkl"
        #style_gan_models[category] = style_gan.get_models(style_gan_network_path)

        # predictor
        predictor_model_path = f'./predictor/network/{category}.pth'
        init_mesh_path = f"./predictor/samples/{samples_per_categories[category]}.obj"
        predictor_model, diffRenderer = predictor.get_predictor_model(init_mesh_path,predictor_model_path,image_size)
        predictor_models[category] = predictor_model
        if samples_per_categories[category] not in diffRenderers:
            diffRenderers[samples_per_categories[category]] = diffRenderer

    # mask 모델 불러오기
    #mask_weights_path = "./mask/weight/mask_rcnn_matilda_0110.h5"
    #mask_model = mask.get_mask_model(mask_weights_path)

    return predictor_models, diffRenderers

def load_cameras_info(root):
    cameras_info = {}
    for category in categories:
        cameras_info[category] = np.load(f'{root}{category}.npy')
    return cameras_info

predictor_models, diffRenderers = get_all_models()

# 카메라 정보 불러오기
#cameras_info = load_cameras_info('./predictor/samples/')

def save_file_into_store(path):
    return

def load_into_numpy_array_and_resize(data, resolution):
    img = Image.open(BytesIO(data))
    W, H = img.size
    desired_size = max(W, H)
    delta_w = desired_size - W
    delta_h = desired_size - H
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding)
    img = img.resize((resolution, resolution), Image.LANCZOS)

    img = np.array(img, dtype=np.uint8)
    return img

@app.get("/")
async def root():
    return {"Welcome"}

@app.post("/convert")
async def convert(file: UploadFile = File(...), category : str = Form(...)):
    image = load_into_numpy_array_and_resize(await file.read(),512)

    predictor = predictor_models[category]
    dib_r = diffRenderers[samples_per_categories[category]]

    ''' in predictor.py '''
    # image를 넣어 mesh, texture 생성
    attributes = predictor(torch.Tensor(image.transpose(2,0,1)).unsqueeze(0).cuda())

    mesh = attributes['vertices']
    texture = attributes['textures']
    #lights = attributes['lights']

    # ''' in style_gan.py '''
    # # mesh, texture를 style gan network에 넣어 다각도 이미지 생성
    # mv_images = style_gan.get_multiview_images(style_gan_models[category], torch.Tensor(cameras_info[category]), mesh, texture)
    #
    # # TODO: mask_rcnn을 사용할 지, IS_NET을 사용할 지 결정
    # ''' in mask.py '''
    # # 이미지들의 sementic mask 얻기
    # mv_masks = mask.detect_mask(mask_model,[mv_images])
    #
    ''' in predictor.py '''
    # 3D Object 생성
    # obj_dir_path, thumb_nail_img = dib_r.create_3d_object(mesh, texture, mv_images, mv_masks, cameras_info[category], category)

    # For Test
    cam_pos = torch.tensor([[0., 0., 6.]], dtype=torch.float).cuda()
    obj_dir_path, thumb_nail_img = dib_r.create_3d_object_not_train(mesh, texture, cam_pos, category)

    cv2.imwrite(f"{category}_thumbnail.png", thumb_nail_img[0].permute(1,2,0).cpu().detach().numpy())

    # # 3D Object를 저장소에 저장
    # save_file_into_store(bin_path)
    # save_file_into_store(obj_path)

    return {"thumb_nail_img": thumb_nail_img.shape, "obj_file_path" : obj_dir_path}
