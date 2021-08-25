#/DeepFashion/In-shop/img/WOMEN/Sweatshirts_Hoodies/id_00004591/01_3_back.jpgCloth_segmented.jpg
from PIL import Image
import os.path


wantedSize = 256
image = Image.open('/data0/yugyoung/VTON/segmented.png')
print(image.filename)
print(image.size)

# 좌우 크기 중 큰 부분 획득
if image.size[0] > image.size[1]:
    tempsize =image.size[0]

else:
    tempsize =image.size[1]

# 변경할 비율 획득
percent = wantedSize/tempsize

# 획득한 비율 출력
print(percent)	

# 획득한 비율을 토대로 이미지 크기 변경 
image=image.resize((int(image.size[0]*percent), int(image.size[1]*percent)))
#image=image.resize((256, 256))
image = image.convert("RGB")
image.save('success.jpg')
print(image.size)
