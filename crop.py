import numpy as np
from numpy import load
from scipy.sparse import csc_matrix
import matplotlib
import matplotlib.pyplot as plt
import cv2


def read_lines(path):
    with open(path) as fin:
        lines = fin.readlines()[2:] #안내문 제거
        lines = list(filter(lambda x: len(x) > 0, lines)) #text가 있으면
        pairs = list(map(lambda x: x.strip().split(), lines)) #글 정제
        #print('success')
    return pairs

def readcloth(path):
    lines = read_lines(path)
    names = set(list(map(lambda x: x[0], lines))) # frontal view
    npz_list=[]
    for name in names:
        name = "/data0/yugyoung/VTON/DeepFashion/In-shop/" + name
        npz_list.append(name)
    return npz_list

def crop_cloth(npz_file):
    
    img_color = cv2.imread(npz_file) # 이미지 파일을 컬러로 불러옴
    img_seg = cv2.imread(npz_file + "seg_.jpg")
    height, width = img_color.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

    img_hsv = cv2.cvtColor(img_seg, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환

    lower_blue = (-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (10, 255, 255)
    img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색
    img_mask = cv2.bitwise_not(img_mask) #흰,검 반대로

    img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)
    #cv2.imwrite(npz_file, img_color)
    npz_file = npz_file.replace(".jpg","_cropmask.jpg")
    cv2.imwrite(npz_file,img_result)
    print('filename: ',npz_file)


def preprocess_data(path):
    npz_list = readcloth(path)
    for npz_f in npz_list:
        crop_cloth(npz_f)

if __name__ == '__main__':
    preprocess_data('/data0/yugyoung/VTON/DeepFashion/In-shop/Anno/list_bbox_inshop.txt') #5~10분