from __future__ import print_function
import torch, gc
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from config import *
from utils import *
from data_loader_seg import Fashion_inshop
from model2 import *
import matplotlib.pyplot as plt
import cv2
from torchsummary import summary
import itertools
from PIL import Image

G_warping = nn.DataParallel(GeneratorResNet(input_shape = (1,6,256,256))).cuda()
G_seg = nn.DataParallel(GeneratorResNet(input_shape = (1,6,256,256))).cuda() #nn.DataParallel(UNet(n_channels=3, n_classes=1, bilinear=True).cuda())
D_seg = nn.DataParallel(Discriminator(input_shape = (1,3,256,256))).cuda()

G_warping.load_state_dict(torch.load("models/G_warping.pt"))
G_seg.load_state_dict(torch.load("models/G_seg_100.pt"))
D_seg.load_state_dict(torch.load("models/D_seg_100.pt"))

G_warping.eval()
G_seg.eval()


data_transform_train = transforms.Compose([
    transforms.Scale(IMG_SIZE),
    transforms.ToTensor(),
    ])

with Image.open('/data0/yugyoung/VTON/DeepFashion/In-shop/img/WOMEN/Cardigans/id_00000119/03_4_full.jpg') as img:
    img = img.convert('RGB')
with Image.open('/data0/yugyoung/VTON/DeepFashion/In-shop/img/WOMEN/Cardigans/id_00006845/01_2_side.jpg') as img_target:
    img_target = img_target.convert('RGB')
with Image.open('/data0/yugyoung/VTON/DeepFashion/In-shop/img/WOMEN/Cardigans/id_00006845/01_2_side.jpgseg_.jpg') as img_tseg:
    img_tseg = img_tseg.convert('RGB')

img = data_transform_train(img)
img_target = data_transform_train(img_target)
img_tseg = data_transform_train(img_tseg)

#-----------test
body = torch.tensor(img, dtype=torch.float32).cuda()
body = body.unsqueeze(0)
print('body.shape: ',body.shape)
tbody = torch.tensor(img, dtype=torch.float32).cuda()
tbody = tbody.unsqueeze(0)
tbody_seg = torch.tensor(img_tseg, dtype = torch.float32).cuda()
tbody_seg = tbody_seg.unsqueeze(0)
body_tbody_seg = torch.cat([body, tbody_seg], dim=1).cuda()
print('success')
fake_body = G_seg(body_tbody_seg).cuda()

G1_tbody = torch.cat([fake_body, tbody], dim=1).cuda()
warping_G1_tbody = G_warping(G1_tbody)

input = torch.cat([body, tbody_seg], dim=0).cuda()
output = torch.cat([input, fake_body], dim=0) #옷: body, 몸: body
row = torch.cat([output, warping_G1_tbody], dim=0)
row_2 = torch.cat([row, tbody], dim=0)
save_out = "test_results_100_5.jpg"

save_image(row_2, save_out, padding=0)
