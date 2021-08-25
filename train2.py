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
from data_loader import Fashion_attr_prediction, Fashion_inshop
from model2 import *
import matplotlib.pyplot as plt
import cv2
from torchsummary import summary
import itertools

data_transform_train = transforms.Compose([
    transforms.Scale(IMG_SIZE),
    transforms.RandomSizedCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#in-shop
train_loader = torch.utils.data.DataLoader(
    Fashion_inshop(type="train", transform=data_transform_train),
    batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    Fashion_inshop(type="test", transform=data_transform_train),
    batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

G_CB = nn.DataParallel(GeneratorResNet(input_shape = (TRAIN_BATCH_SIZE, 6,256,256))).cuda()
D_CB = nn.DataParallel(Discriminator(input_shape = (TRAIN_BATCH_SIZE, 3,256,256))).cuda()

#Buffers of previously generated samples
fake_CB_buffer = ReplayBuffer()

gc.collect()
torch.cuda.empty_cache() 

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#Optimizers
optimizer_G_CB = optim.SGD(filter(lambda p: p.requires_grad, G_CB.parameters()), lr=LR, momentum=MOMENTUM)
optimizer_D_CB = torch.optim.Adam(D_CB.parameters(), lr=LR)

# Learning rate update schedulers
lr_scheduler_G_CB = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G_CB, lr_lambda=LambdaLR(EPOCH, 0,  EPOCH*0.5).step
)
lr_scheduler_D_CB = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_CB, lr_lambda=LambdaLR(EPOCH, 0, EPOCH*0.5).step
)
'''fake_CB.shape:  torch.Size([4, 3, 256, 256])
fCB_cloth.shape:  torch.Size([4, 6, 256, 256])
tcloth_recv.shape:  torch.Size([4, 3, 256, 256])'''

if __name__ == '__main__':
    G_CB.train()
    
    prev_time = time.time()
    for epoch in range(1, EPOCH +1):
        for (batch_idx, target) in enumerate(train_loader): 
            #print("target.size: ",target.size)
            save_inp = "results/210728/inputs"
            save_out = "results/210728/outputs"
            cloth = torch.tensor(target[0], dtype=torch.float32).cuda()
            body = torch.tensor(target[1], dtype=torch.float32).cuda()
            #print('target_cloth.shape: ',target_cloth.shape)
            #print('target_body.shape: ',target_body.shape)
                        
            cloth_body = torch.cat([cloth, body], dim=1).cuda()
            #print('cloth_body.shape: ',cloth_body.shape)

            _cloth_body = torch.cat([cloth, body], dim=0)
            save_inp = save_inp+str(batch_idx)+".jpg"
            save_image(_cloth_body, save_inp)


            fake_CB = G_CB.forward(cloth_body).cuda()
            #------------------stage_one_end
            #------------------stage_two_start
            
            
            fCB_cloth = torch.cat([fake_CB, cloth], dim=1)
            #print("fCB_cloth.shape: ", fCB_cloth.shape)
            tcloth_recv = G_CB.forward(fCB_cloth).cuda()
            #print("tcloth_rcv.shape: ", tcloth_recv.shape)
            #stage_two_end
            
            #G loss
            optimizer_G_CB.zero_grad()
            
            # Identity loss
            loss_identity = criterion_identity(tcloth_recv, cloth) #L1 loss
            
            # GAN loss = BCEWithLogitsLoss
            valid = torch.tensor(np.ones((cloth.size(0), *D_CB.module.output_shape)), requires_grad=False, dtype=torch.float32).cuda()
            loss_GAN = criterion_GAN(D_CB(tcloth_recv), valid)
            # Total loss
            loss_G_CB = loss_GAN + LAMBDA_ID * loss_identity

            loss_G_CB.backward(retain_graph=True)
            optimizer_G_CB.step()
                        
            #out  = outputs.numpy()
            #out = outputs[0].detach().cpu().clone().numpy().transpose(0,1,2)
            #print('output shape: ',fake_CB.shape)
            save_out = save_out+str(batch_idx)+".jpg"
            save_image(tcloth_recv, save_out)
            
            
            # D loss
            optimizer_D_CB.zero_grad()
            pred_real_cloth = D_CB(cloth)
            pred_fake_cloth = D_CB(tcloth_recv)
            
            loss_D_CB_real = criterion_GAN(pred_real_cloth, torch.ones_like(pred_real_cloth))
            loss_D_CB_fake = criterion_GAN(pred_fake_cloth, torch.zeros_like(pred_fake_cloth))
            loss_D_CB = 0.5 * (loss_D_CB_real + loss_D_CB_fake)

            loss_D_CB.backward(retain_graph=True)
            optimizer_D_CB.step()
            

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + batch_idx
            batches_left = EPOCH * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    EPOCH,
                    batch_idx,
                    len(train_loader),
                    loss_D_CB.item(),
                    loss_G_CB.item(),
                    loss_GAN.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            #if batches_done % SAMPLE_INTERVAL == 0:
                #sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G_CB.step()
        lr_scheduler_D_CB.step()

        print("sucess")
        