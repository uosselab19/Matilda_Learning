# Matilda Machine Learning
* * *
<b> Matilda can procede 3d object by only one image !! </b>
<br>
we provice categories 

+ DR : dress
+ TOP : shirts
+ BTM : pants
+ HEA : hat
+ BRA : bra(?)
+ NEC : neckless
+ BAG : bag 
+ MAS : mask 
+ RIN : ring

## Required libraries
* * *
+ pip install -r requirements.txt 
+ in requirements.txt, you can look these libraries 

Python 3.7\
fastapi \
uvicorn[standard] \
ipdb\
trimesh\
numpy\
torch==1.7.1\
torchvision==0.8.2\
torchaudio==0.7.2\
requests\
scikit-image\
matplotlib\
cython==0.29.20\
usd-core==22.3\
aspose-3d\
opencv-python\
python-multipart\
jwt\
PyJWT\
httpx\

## Please Clone Extra Loss Function 
* * *
git clone https://github.com/shubhtuls/PerceptualSimilarity.git 

in PerceptualSimilarity.util.util, change code
```
from skimage.measure import compare_ssim
```
to
```
from skimage.metrics import structural_similarity as compare_ssim
```

## Examples
* * *