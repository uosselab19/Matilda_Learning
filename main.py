import sys
from fastapi import FastAPI, File, Form, UploadFile, Header
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import os
import httpx
import jwt
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

#from style_gan import style_gan
from mask import mask
from predictor import predictor

# start : uvicorn main:app --host 0.0.0.0 --port 8100 --reload &
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://3.133.233.81:3000"
    "https://localhost:3000",
    "https://3.133.233.81:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image Storage 위치 지정 및 디렉토리 확인
PATH = os.environ.get("ImageStorage") + '/items'

if not os.path.exists(PATH):
    print("지정된 Image Storage 경로를 찾지 못했습니다.")
    print("프로그램을 종료합니다.")
    exit()

categories = ['TOP']
# categories = ['DR', 'TOP', 'BTM', 'HEA', 'BRA', 'NEC', 'BAG', 'MAS', 'RIN']
for cate in categories:
    if not os.path.exists(os.path.join(PATH, cate)):
        print("지정된 Image Storage에서 다음 카테고리에 해당하는 디렉토리를 찾지 못했습니다.: " + cate)
        print("프로그램을 종료합니다.")
        exit()

samples_per_categories = {'DR': 'sphere','TOP': 'sphere', 'BTM': 'sphere', 'HEA': 'sphere', 'NEC': 'torus', 'BAG': 'torus', 'MAS':'sphere', 'RIN':'torus'}

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
    model_path = "./mask/DIS/saved_models/isnet.pth"  ## load trained weights from this path
    #mask_weights_path = "./mask/weight/mask_rcnn_matilda_0110.h5"
    mask_model = mask.get_mask_model(model_path)

    return predictor_models, mask_model, diffRenderers

def load_cameras_info(root):
    cameras_info = {}
    for category in categories:
        cameras_info[category] = np.load(f'{root}{category}.npy')
    return cameras_info

''' 모델 및 카메라 정보 가져오기'''
predictor_models, mask_model, diffRenderers = get_all_models()

# 카메라 정보 불러오기
#cameras_info = load_cameras_info('./predictor/samples/')

def load_into_tensor_and_resize(data, resolution, mask_model):
    img = Image.open(BytesIO(data)).convert('RGB')

    W, H = img.size
    desired_size = max(W, H)
    delta_w = desired_size - W
    delta_h = desired_size - H
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding)

    img = img.resize((resolution, resolution))

    # tf = transforms.Compose([transforms.Resize(resolution),
    #                          transforms.CenterCrop(resolution)])
    # img = tf(img)
    img = torchvision.transforms.functional.to_tensor(img).cuda()

    img_mask = mask.get_mask_from_image(mask_model, img.unsqueeze(0))
    img_mask = torch.where(img_mask > 0.6, 1., 0.)

    img = img * img_mask + torch.ones_like(img) * (1 - img_mask)

    return img

# WAS를 통해 Repository에 파일 저장
URL = "http://3.133.233.81:8080"
def save_file_into_repository(title: str, catCode: str, saveUrl: str, token: str):
    memberNum = jwt.decode(token, options={"verify_signature": False})['num']
    body = {
        "title": title,
        "catCode": catCode,
        "imgUrl": saveUrl + '/thumbImg.jpg',
        "memberNum": memberNum,
        "objectUrl": saveUrl
    }
    response = httpx.post(URL + '/items/new', json=body)
    return response.json()

@app.get("/")
async def root():
    return {"Welcome"}

@app.post("/convert")
async def convert(file: UploadFile = File(...), category : str = Form(...), X_AUTH_TOKEN: str = Header()):
    # 카테고리가 유효한지 확인
    if category not in categories:
        return {"message": "category not found"}
    
    # 파일 이름 추출
    title = file.filename
    if len(title) > 45:
        title = title[0:45]
    
    image = load_into_tensor_and_resize(await file.read(),512, mask_model) # image 사이즈 조절 및 tensor로 변환

    predictor = predictor_models[category] # category에 해당하는 3D 속성 예측 모델 불러오기
    dib_r = diffRenderers[samples_per_categories[category]] # category에 해당하는 3D Renderer 불러오기

    ''' in predictor.py '''
    # image를 넣어 mesh, texture, lights 생성
    attributes = predictor(image.unsqueeze(0))

    mesh = attributes['vertices']
    texture = attributes['textures']
    lights = attributes['lights']

    # 파일 저장되는 위치
    dirName = category + '/' + str(int(datetime.now().timestamp())) + '_' + title
    saveUrl = os.path.join(PATH, dirName)

    # 3D Object 생성 - 생성된 mesh, texture, lights를 통해 3D 파일(.glb) 추출하기
    objUrl = dib_r.save_object(mesh, texture, lights, category, saveUrl)

    # WAS로 saveUrl 전달
    response = save_file_into_repository(title, category, os.path.join('items', dirName), X_AUTH_TOKEN)

    return response
