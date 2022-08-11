import os
import httpx
import jwt
from fastapi import FastAPI, File, Form, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# start : uvicorn main:app --reload
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
path = os.environ.get("ImageStorage") + '\\items'

if not os.path.exists(path):
    print("지정된 Image Storage 경로를 찾지 못했습니다.")
    print("프로그램을 종료합니다.")
    exit()

categories = ['DR', 'TOP', 'BTM', 'HEA', 'BRA', 'NEC', 'BAG', 'MAS', 'RIN']
for cate in categories:
    if not os.path.exists(os.path.join(path, cate)):
        print("지정된 Image Storage에서 다음 카테고리에 해당하는 디렉토리를 찾지 못했습니다.: " + cate)
        print("프로그램을 종료합니다.")
        exit()


# Image Storage에 gltf, bin, thumbImg를 저장
def save_file_into_storage(title: str, catCode: str, gltf: bytes, bin: bytes, thumbImg: bytes):
    dirName = catCode + '\\' + str(int(datetime.now().timestamp())) + '_' + title
    saveUrl = os.path.join(path, dirName)
    os.makedirs(saveUrl)

    with open(os.path.join(saveUrl, '2CylinderEngine.gltf'), "wb") as f:
        f.write(gltf)
    with open(os.path.join(saveUrl, '2CylinderEngine.bin'), "wb") as f:
        f.write(bin)
    with open(os.path.join(saveUrl, 'thumbImg.jpg'), "wb") as f:
        f.write(thumbImg)

    return dirName


# WAS를 통해 Repository에 파일 저장
URL = "http://localhost:8080"
def save_file_into_repository(title: str, catCode: str, saveUrl: str, token: str):
    memberNum = jwt.decode(token, options={"verify_signature": False})['num']
    body = {
        "title": title,
        "catCode": catCode,
        "imgUrl": saveUrl + '\\thumbImg.jpg',
        "memberNum": memberNum,
        "objectUrl": saveUrl
    }
    response = httpx.post(URL + '/items/new', json=body)
    return response.json()

# 메인 페이지 접속
@app.get("/")
async def root():
    return {"Welcome"}

# 3D Conversion 수행
@app.post("/convert")
async def convert(file: UploadFile = File(...), catCode: str = Form(...), X_AUTH_TOKEN: str = Header()):
    # 카테고리가 유효한지 확인
    if catCode not in categories:
        return {"message": "category not found"}

    title = file.filename
    if len(title) > 45:
        title = title[0:45]

    # 3D conversion 수행
    bin_file = open("data\\2CylinderEngine.bin", 'rb').read()
    obj_file = open("data\\2CylinderEngine.gltf", 'rb').read()
    thumbImg_file = await file.read()

    # 변환된 파일 저장
    saveUrl = save_file_into_storage(title, catCode, obj_file, bin_file, thumbImg_file)

    # WAS로 saveUrl 전달
    response = save_file_into_repository(title, catCode, saveUrl, X_AUTH_TOKEN)

    # 변환 및 저장 완료 결과 반환
    return response


@app.get("/test")
async def test():
    response = httpx.get(URL + '/items')
    return response.json()
