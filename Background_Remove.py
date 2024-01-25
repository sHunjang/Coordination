from rembg import remove
from PIL import Image

TOP_img_dir = '/Users/seunghunjang/Desktop/Coordination/Coordination/train/TOP'
BOTTOM_img_dir = '/Users/seunghunjang/Desktop/Coordination/Coordination/train/BOTTOM'


# 이미지 배경 제거
input = Image.open('/Users/seunghunjang/Desktop/MNIST_Test/Coordination/train/TOP/TOP_4.jpg') # load image
output = remove(input) # remove background
output.save('TOP_4_B.PNG') # save image