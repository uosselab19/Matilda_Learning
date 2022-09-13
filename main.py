from fastapi import FastAPI, File, Form, UploadFile, Header
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import shutil
import httpx
import jwt
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import boto3
from store_NFTStorage import store_NFTStorage
import io
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

categories = ['TOP', 'BTM']
#categories = ['DR', 'TOP', 'BTM', 'HEA', 'BRA', 'NEC', 'BAG', 'MAS', 'RIN']
samples_per_categories = {'DR': 'sphere','TOP': 'sphere', 'BTM': 'sphere', 'HEA': 'sphere', 'NEC': 'torus', 'BAG': 'torus', 'MAS':'sphere', 'RIN':'torus'}
args_per_categories = {
    'TOP': {
        'azi_scope' : 180,
        'elev_range' : '-30~30',
        'dist_range' : '5~7'
    },
    'BTM': {
        'azi_scope': 180,
        'elev_range': '-30~30',
        'dist_range': '5~7'
    }
}

def get_all_models():
    image_size = 512

    predictor_models = {}
    diffRenderers = {}

    origin = "/home/ec2-user/Matilda_Learning"

    for category in categories:
        # predictor
        predictor_model_path = f'{origin}/predictor/network/models/{category}.pth'
        init_mesh_path = f"{origin}/predictor/samples/{samples_per_categories[category]}.obj"
        predictor_model, diffRenderer = predictor.get_predictor_model(init_mesh_path,predictor_model_path,image_size,args_per_categories[category])
        predictor_models[category] = predictor_model
        if samples_per_categories[category] not in diffRenderers:
            diffRenderers[samples_per_categories[category]] = diffRenderer

    # mask 모델 불러오기
    model_path = f"{origin}/mask/DIS/saved_models/isnet.pth"  ## load trained weights from this path
    mask_model = mask.get_mask_model(model_path)

    return predictor_models, mask_model, diffRenderers

''' 모델 및 카메라 정보 가져오기'''
predictor_models, mask_model, diffRenderers = get_all_models()

# WAS를 통해 Repository에 파일 저장
URL = "http://3.133.233.81:8080"
def save_fileinfo_into_repository(title: str, catCode: str, imgUrl: str, objectUrl: str, token: str) -> str:
    memberNum = jwt.decode(token, options={"verify_signature": False})['num']
    body = {
        "title": title,
        "catCode": catCode,
        "imgUrl": imgUrl,
        "memberNum": memberNum,
        "objectUrl": objectUrl
    }
    response = httpx.post(URL + '/items/new', json=body)
    return response.json()

def get_fileinfo_from_repository(num: int, token: str) -> str:
    response = httpx.get(URL + '/objects/auth/objUrl/' +
                         str(num), headers={'X-AUTH-TOKEN': token})

    return response.text

bucketMatilda = boto3.resource('s3').Bucket('matilda.image-storage')

def save_file_into_S3(localfilePath: str, targetfilePath: str):

    bucketMatilda.upload_file(localfilePath, targetfilePath)

    return True

def get_file_from_S3(filePath: str) -> io.BytesIO:
    fileData = io.BytesIO()

    bucketMatilda.download_fileobj(filePath, fileData)
    fileData.seek(0)

    return fileData

def load_into_tensor_and_resize(data, resolution, mask_model):
    img = Image.open(BytesIO(data)).convert('RGB')

    tf = transforms.Compose([transforms.Resize(resolution),
                             transforms.CenterCrop(resolution)])
    img = tf(img)
    img = torchvision.transforms.functional.to_tensor(img).cuda()
    img_mask = mask.get_mask_from_image(mask_model, img.unsqueeze(0))

    print(img_mask.shape)
    img = img * img_mask + torch.ones_like(img) * (1 - img_mask.unsqueeze(0))

    return img

@app.get("/")
async def root():
    return {"Welcome"}

@app.post("/convert")
async def convert(file: UploadFile = File(...), category: str = Form(...), X_AUTH_TOKEN: str = Header()):
    # 카테고리가 유효한지 확인
    if category not in categories:
        return {"message": "category not found"}

    # 파일 이름 추출
    title = '.'.join(file.filename.split('.')[:-1])
    if len(title) > 45:
        title = title[0:45]

    image = load_into_tensor_and_resize(await file.read(),512, mask_model) # image 사이즈 조절 및 tensor로 변환

    predictor = predictor_models[category] # category에 해당하는 3D 속성 예측 모델 불러오기
    dib_r = diffRenderers[samples_per_categories[category]] # category에 해당하는 3D Renderer 불러오기

    attributes = predictor(image.unsqueeze(0))
    
    # 파일 이름에 사용 할 시간 정보
    now = str(int(datetime.now().timestamp()))

    # 파일이 로컬에 임시로 저장될 위치
    save_path = './temp/' + now + '/'

    # 3D Object 생성 - 생성된 mesh, texture, lights를 통해 3D 파일(.glb) 추출하기
    obj_save_path, img_save_path = dib_r.save_object(attributes, category, save_path)
    
    # 파일이 S3에 저장될 위치
    objPath = 'items/obj/' + category + '/' + now + '_' + title + '.glb'
    imgPath = 'items/img/' + category + '/' + now + '_' + title + '.jpg'

    # S3에 파일 저장
    save_file_into_S3(obj_save_path, objPath)
    save_file_into_S3(img_save_path, imgPath)

    # 로컬 파일 삭제
    shutil.rmtree(save_path)

    # WAS로 saveUrl 전달
    response = save_fileinfo_into_repository(
        title, category, imgPath, objPath, X_AUTH_TOKEN)

    return response


@app.post("/getCID")
async def getCID(num: int = Form(...), X_AUTH_TOKEN: str = Header()):
    # 유효 확인

    # WAS를 통해 파일 정보 호출
    filePath = get_fileinfo_from_repository(num, X_AUTH_TOKEN)

    # S3로부터 파일 다운로드
    fileData = get_file_from_S3(filePath)

    # NFT.Storage에 파일 저장, CID 획득
    cid = store_NFTStorage(fileData)

    # FE로 cid 정보 반환
    return cid
