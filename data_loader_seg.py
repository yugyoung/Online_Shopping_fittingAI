#ref: https://github.com/ihciah/deep-fashion-retrieval/blob/master/data.py
import torch.utils.data as data
import torch
from config import *
import os
from PIL import Image
import random

class Fashion_inshop(data.Dataset):
    def __init__(self, type="train", transform=None):
        self.transform = transform
        self.type = type
        self.train_dict = {}
        self.test_dict = {}
        self.train_list = []
        self.test_list = []
        self.all_path = []
        self.cloth = self.readcloth()
        self.read_train_test()
        #print("self.train_dict: ",self.train_dict) #cloth type list
        #print('self.train_list: ',self.train_list) #specific id list
    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:] #안내문 제거
            lines = list(filter(lambda x: len(x) > 0, lines)) #text가 있으면
            pairs = list(map(lambda x: x.strip().split(), lines)) #글 정제
        return pairs

    def readcloth(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'In-shop', 'Anno/list_bbox_inshop.txt'))
        valid_lines = list(filter(lambda x: x[1] == '1' and (x[2] =='1'or x[2]=='2'), lines)) #upper-body clothes & frontal view
        #valid_lines_side = list(filter(lambda x: x[1] == '1' and x[2] =='2', lines)) #upper-body clothes & side view
        with open("train_list.txt","w") as f:
            f.write(str(valid_lines))
            f.write('\n')
        names = set(list(map(lambda x: x[0], valid_lines))) # frontal view
        #names_seg = set(list(map(lambda x: x[0]+'seg_.jpg', valid_lines))) # frontal view
        #names_side = set(list(map(lambda x: x[0], valid_lines))) # frontal view
        return names

    def read_train_test(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'In-shop', 'Eval/list_eval_partition.txt'))
        valid_lines = list(filter(lambda x: x[0] in self.cloth, lines)) #Eval에서 front(train시킬것)에 해당하는 것만 빼와.
        #valid_seg_lines = list(filter(lambda x: x[0] in self.cloth_side, lines))
        for line in valid_lines:
            s = self.train_dict if line[2] == 'train' else self.test_dict
            if line[1] not in s:
                s[line[1]] = [line[0]]
            else:
                s[line[1]].append(line[0])
        
        def clear_single(d):
            keys_to_delete = []
            for k, v in d.items():
                if len(v) < 2:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                d.pop(k, None)
        clear_single(self.train_dict)
        clear_single(self.test_dict)
        self.train_list, self.test_list = list(self.train_dict.keys()), list(self.test_dict.keys())
        for v in list(self.train_dict.values()):
            self.all_path += v
        self.train_len = len(self.all_path)
        for v in list(self.test_dict.values()):
            self.all_path += v
        self.test_len = len(self.all_path) - self.train_len

    def process_img(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, 'In-shop', img_path)
        with open(img_full_path, 'rb') as f:
            #img = cv2.imread(f)
            #print('f: ',f)
            with Image.open(f) as img:
                img = img.convert('RGB')
            n = f.name
            n = n + 'seg_.jpg'
            with Image.open(n) as img_seg:
                #print(img_seg)
                img_seg = img_seg.convert('RGB')
                
        if self.transform is not None:
            img = self.transform(img)
            img_seg = self.transform(img_seg)
        #print('hererererere',img.shape) #3, 224, 224
        return (img, img_seg) #tuple

    def __len__(self):
        if self.type == 'train':
            return len(self.train_list)
        elif self.type == 'test':
            return len(self.test_list)
        else:
            return len(self.all_path)

    def __getitem__(self, item):
        if self.type == 'all':
            img_path = self.all_path[item]
            img, img_seg = self.process_img(img_path)
            return img,img_path, img_path
        s_d = self.train_dict if self.type == 'train' else self.test_dict #self.train_dict[id_00000005] = 'img/WOMEN/~/01_1_fron.jpg'
        s_l = self.train_list if self.type == 'train' else self.test_list #self.train_list = id_0000005,id_00000006,~
        imgs = s_d[s_l[item]]
        #print('imgs: ',imgs)#id 하나의 카테고리
        while len(imgs)<2:
            imgs = s_d[s_l[item]]
        img_triplet = []
        img_triplet.append(imgs[0])
        img_triplet.append(imgs[1]) #data augmentation이 있음(front-side pair가 안 맞는게 있음)
        #print('img_triplet: ',img_triplet)
        #print('img_triplet: ',img_triplet) #imgs: ['img/WOMEN/Sweaters/id_00000039/03_1_front.jpg', 'img/WOMEN/Sweaters/id_00000039/03_2_side.jpg', 'img/WOMEN/Sweaters/id_00000039/04_1_front.jpg', 'img/WOMEN/Sweaters/id_00000039/04_2_side.jpg']
        #img_other_id = random.choice(list(range(0, item)) + list(range(item + 1, len(s_l))))
        #img_other = random.choice(s_d[s_l[img_other_id]])
        #img_triplet.append(img_other)
        return list(map(self.process_img, img_triplet)) #img 주소 하나씩 들어가