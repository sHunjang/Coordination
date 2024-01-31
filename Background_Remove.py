from rembg import remove
from PIL import Image



# 이미지 배경 제거
input = Image.open('/Users/seunghunjang/Desktop/Coordination/Coordination/train/BOTTOM/BOTTOM_5.jpg') # load image
output = remove(input) # remove background
output.save('BOTTOM_5_B.PNG') # save image