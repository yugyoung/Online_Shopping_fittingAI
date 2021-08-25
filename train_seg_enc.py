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
from model3 import *
import matplotlib.pyplot as plt
import cv2
from torchsummary import summary
import itertools

data_transform_train = transforms.Compose([
    transforms.Scale(IMG_SIZE),
    transforms.ToTensor(),
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

Feature_enc = nn.DataParallel(CocosEnc()).cuda()
Feature_seg_enc = nn.DataParallel(CocosEnc()).cuda()
G_seg = nn.DataParallel(GeneratorResNet(input_shape = (TRAIN_BATCH_SIZE, 6,256,256))).cuda() #nn.DataParallel(UNet(n_channels=3, n_classes=1, bilinear=True).cuda())
D_seg = nn.DataParallel(Discriminator(input_shape = (TRAIN_BATCH_SIZE, 3,256,256))).cuda() #nn.DataParallel(MultiscaleDiscriminator(input_nc = 3).cuda())
#Buffers of previously generated samples
fake_CB_buffer = ReplayBuffer()

gc.collect()
torch.cuda.empty_cache() 

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#Optimizers
optimizer_G_seg = torch.optim.Adam(
    itertools.chain(G_seg.parameters(), G_seg.parameters()), lr=LR
)
optimizer_D_seg = torch.optim.Adam(D_seg.parameters(), lr=LR)

# Learning rate update schedulers
lr_scheduler_G_seg = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G_seg, lr_lambda=LambdaLR(EPOCH, 0,  EPOCH*0.5).step
)
lr_scheduler_D_seg = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_seg, lr_lambda=LambdaLR(EPOCH, 0, EPOCH*0.5).step
)

if __name__ == '__main__':
    prev_time = time.time()
    G_seg.train()
    Feature_enc.train()
    for epoch in range(1, EPOCH +1):
        for (batch_idx, target) in enumerate(train_loader): 
            #print("target.size: ",target.size)
            save_out = "results/train_seg_20/"
            #print('target.shape: ',len(target))
            body = torch.tensor(target[0][0], dtype=torch.float32).cuda()
            #body_seg = torch.tensor(target[0][1], dtype = torch.float32).cuda() #cycle_loss 줄 때 사용
            tbody = torch.tensor(target[1][0], dtype=torch.float32).cuda()
            tbody_seg = torch.tensor(target[1][1], dtype = torch.float32).cuda()

            body_feature = Feature_enc(body) #[4, 64, 59, 59]
            body_seg_feature = Feature_seg_enc(tbody_seg) #[4, 64, 59, 59]



            body_tbody_seg = torch.cat([body_feature, body_seg_feature], dim=1).cuda()
            print('body_tbody_seg.shape: ',body_tbody_seg.shape)# [4, 128, 59, 59]
            fake_body = G_seg.forward(body_tbody_seg).cuda() #([4, 3, 56, 56])
            print('fake_body.shape: ',fake_body.shape)
            #------------------stage_one_end

            #G loss
            optimizer_G_seg.zero_grad()
            
            # Identity loss
            loss_identity_CB = criterion_identity(fake_body, tbody) #L1 loss
            #loss_identity_CB += criterion_identity(tcloth_recv_2, cloth) #L1 loss
            #loss_identity_CB /= 2 #loss_identity = (loss_id_A + loss_id_B) / 2
            
            loss_identity = (loss_identity_CB )
            
            # GAN loss = BCEWithLogitsLoss
            valid = torch.tensor(np.ones((tbody.size(0), *D_seg.module.output_shape)), requires_grad=False, dtype=torch.float32).cuda()
            loss_GAN_CB = criterion_GAN(D_seg(fake_body), valid)
            loss_GAN_CB += criterion_GAN(D_seg(fake_body), valid)
            loss_GAN_CB /= 2 #loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            loss_GAN = (loss_GAN_CB )

            #Cycle Loss
            '''loss_cycle_CB = criterion_cycle(cycle_fake_CB, cloth) #L1Loss
            loss_cycle_BC = criterion_cycle(cycle_fake_BC, body) #L1Loss

            loss_cycle = (loss_cycle_CB + loss_cycle_BC) / 2
            '''
            # Total loss
            loss_G = loss_GAN + LAMBDA_ID * loss_identity

            loss_G.backward(retain_graph=True)
            optimizer_G_seg.step()
                        
            #out  = outputs.numpy()
            #out = outputs[0].detach().cpu().clone().numpy().transpose(0,1,2)
            #print('output shape: ',fake_CB.shape)

            input = torch.cat([body, tbody_seg], dim=0).cuda()
            output = torch.cat([input, fake_body], dim=0) #옷: body, 몸: body
            row = torch.cat([output, tbody], dim=0) #옷: body, 몸: body
            #tbody_recv = G_CB.forward(fBC_body).cuda()
            #tbody_recv_2 = G_CB.forward(fCB_body).cuda()
            #_G_CB_output = torch.cat([G_CB_output, tbody_recv, tbody_recv_2], dim=0).cuda()
            #save_inp_oup = torch.cat([_G_CB_output, fake_CB, fake_BC, _cloth_body], dim=0).cuda()
            save_out = save_out+str(batch_idx)+".jpg"
            save_image(row, save_out, padding=0)
            
            
            # D_CB loss
            optimizer_D_seg.zero_grad()
            pred_real_body = D_seg(tbody)
            pred_fake_body = D_seg(fake_body)
            #pred_fake_cloth_2 = D_seg(tcloth_recv_2)
            
            loss_D_CB_real = criterion_GAN(pred_real_body, torch.ones_like(pred_real_body))
            loss_D_CB_fake = criterion_GAN(pred_fake_body, torch.zeros_like(pred_fake_body))
            #loss_D_CB_fake_2 = criterion_GAN(pred_fake_cloth, torch.zeros_like(pred_fake_cloth))
            loss_D_CB = 0.5 * (loss_D_CB_real + loss_D_CB_fake) #+ 0.5 * (loss_D_CB_real + loss_D_CB_fake_2)

            loss_D_CB.backward(retain_graph=True)
            optimizer_D_seg.step()
            
            loss_D = (loss_D_CB)

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
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    #loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            #if batches_done % SAMPLE_INTERVAL == 0:
                #sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G_seg.step()
        lr_scheduler_D_seg.step()
        print("sucess")
        