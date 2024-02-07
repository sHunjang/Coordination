from rembg import remove
from PIL import Image



# 이미지 배경 제거
input = Image.open('path/to/img.PNG') # load image
output = remove(input) # remove background
output.save('rembg_img_name.PNG') # save image