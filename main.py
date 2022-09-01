import os
from fastapi import FastAPI, File, Form, UploadFile, Header
# import torchvision
# import torchvision.transforms as transforms
import httpx
import jwt
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import boto3
from store_NFTStorage import store_NFTStorage
import io

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

categories = ['DR', 'TOP', 'BTM', 'HEA', 'BRA', 'NEC', 'BAG', 'MAS', 'RIN']

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

    # 파일 이름에 사용 할 시간 정보
    now = str(int(datetime.now().timestamp()))

    # 파일이 로컬에 임시로 저장될 위치
    obj_save_path = './temp/' + now + '/obj.glb'
    img_save_path = './temp/' + now + '/img.jpg'

    # conversion 수행
    # prediect(obj_save_path, img_save_path, ....)

    # 파일이 S3에 저장될 위치
    objPath = 'items/obj/' + category + '/' + now + '_' + title + '.glb'
    imgPath = 'items/img/' + category + '/' + now + '_' + title + '.jpg'

    # S3에 파일 저장
    save_file_into_S3(obj_save_path, objPath)
    save_file_into_S3(img_save_path, imgPath)

    # 로컬 파일 삭제
    os.remove(obj_save_path)
    os.remove(img_save_path)

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
