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
from data_loader_2 import Fashion_attr_prediction, Fashion_inshop
from model2 import *
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

G_CB = nn.DataParallel(GeneratorResNet(input_shape = (TRAIN_BATCH_SIZE, 6,256,256))).cuda()
D_CB = nn.DataParallel(Discriminator(input_shape = (TRAIN_BATCH_SIZE, 3,256,256))).cuda()
G_BC = nn.DataParallel(GeneratorResNet(input_shape = (TRAIN_BATCH_SIZE, 6,256,256))).cuda()
#Buffers of previously generated samples
fake_CB_buffer = ReplayBuffer()

gc.collect()
torch.cuda.empty_cache() 

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_CB.parameters(), G_BC.parameters()), lr=LR
)
optimizer_D_CB = torch.optim.Adam(D_CB.parameters(), lr=LR)

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(EPOCH, 0,  EPOCH*0.5).step
)
lr_scheduler_D_CB = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_CB, lr_lambda=LambdaLR(EPOCH, 0, EPOCH*0.5).step
)

if __name__ == '__main__':
    prev_time = time.time()
    G_CB.train()
    G_BC.train()
    for epoch in range(1, EPOCH +1):
        for (batch_idx, target) in enumerate(train_loader): 
            #print("target.size: ",target.size)
            save_out = "results/0730/train6_20/outputs"
            cloth = torch.tensor(target[0], dtype=torch.float32).cuda()
            body = torch.tensor(target[1], dtype=torch.float32).cuda()
            #print('target_cloth.shape: ',target_cloth.shape)
            #print('target_body.shape: ',target_body.shape)
                        
            cloth_body = torch.cat([cloth, body], dim=1).cuda()
            body_cloth = torch.cat([body, cloth], dim=1).cuda()
            #print('cloth_body.shape: ',cloth_body.shape)

            _cloth_body = torch.cat([cloth, body], dim=0)
            '''save_inp = save_inp+str(batch_idx)+".jpg"
            save_image(_cloth_body, save_inp)'''


            fake_CB = G_CB.forward(cloth_body).cuda()
            fake_BC = G_CB.forward(body_cloth).cuda()
            #------------------stage_one_end

            #------------------cyclegan stage one
            fcloth_fbody = torch.cat([fake_CB, fake_BC], dim=1).cuda()
            fbody_fcloth = torch.cat([fake_BC, fake_CB], dim=1).cuda()
            cycle_fake_CB = G_BC.forward(fcloth_fbody).cuda()
            cycle_fake_BC = G_BC.forward(fbody_fcloth).cuda()


            #------------------stage_two_start
            fCB_cloth = torch.cat([fake_CB, cloth], dim=1)
            fBC_cloth = torch.cat([cloth, fake_BC], dim=1)
            tcloth_recv = G_CB.forward(fCB_cloth).cuda()
            tcloth_recv_2 = G_CB.forward(fBC_cloth).cuda()

            

            #G loss
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_identity_CB = criterion_identity(tcloth_recv, cloth) #L1 loss
            loss_identity_CB += criterion_identity(tcloth_recv_2, cloth) #L1 loss
            loss_identity_CB /= 2 #loss_identity = (loss_id_A + loss_id_B) / 2
            
            loss_identity = (loss_identity_CB )
            
            # GAN loss = BCEWithLogitsLoss
            valid = torch.tensor(np.ones((cloth.size(0), *D_CB.module.output_shape)), requires_grad=False, dtype=torch.float32).cuda()
            loss_GAN_CB = criterion_GAN(D_CB(tcloth_recv), valid)
            loss_GAN_CB += criterion_GAN(D_CB(tcloth_recv_2), valid)
            loss_GAN_CB /= 2 #loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            loss_GAN = (loss_GAN_CB )

            #Cycle Loss
            recov_cycle_CB = G_BC(cloth_body) #<-> fake_CB = G_CB.forward(cloth_body).cuda()
            recov_cycle_BC = G_BC(body_cloth)
            loss_cycle_CB = criterion_cycle(recov_cycle_CB, cycle_fake_CB) #L1Loss
            loss_cycle_BC = criterion_cycle(recov_cycle_BC, cycle_fake_BC) #L1Loss
            loss_cycle_G_BC = loss_cycle_CB + loss_cycle_BC

            cycle_G_CB_output = torch.cat([cycle_fake_CB, cycle_fake_BC], dim=1)
            cycle_G_BC_output = torch.cat([cycle_fake_BC, cycle_fake_CB], dim=1)
            recov_CB = G_CB(cycle_G_CB_output)
            recov_BC = G_CB(cycle_G_BC_output)
            loss_CB = criterion_cycle(recov_CB, fake_CB)
            loss_BC = criterion_cycle(recov_BC, fake_BC)
            loss_cycle_G_CB = loss_CB + loss_BC

            loss_cycle = (loss_cycle_G_CB + loss_cycle_G_BC) / 2

            # Total loss
            loss_G = loss_GAN + LAMBDA_CYC * loss_cycle + LAMBDA_ID * loss_identity

            loss_G.backward(retain_graph=True)
            optimizer_G.step()
                        
            #out  = outputs.numpy()
            #out = outputs[0].detach().cpu().clone().numpy().transpose(0,1,2)
            #print('output shape: ',fake_CB.shape)

            G_CB_output = torch.cat([tcloth_recv, tcloth_recv_2], dim=0).cuda()
            save_inp_oup = torch.cat([G_CB_output, _cloth_body], dim=0).cuda()
            save_out = save_out+str(batch_idx)+".jpg"
            save_image(save_inp_oup, save_out, padding=0)
            
            
            # D_CB loss
            optimizer_D_CB.zero_grad()
            pred_real_cloth = D_CB(cloth)
            pred_fake_cloth = D_CB(tcloth_recv)
            pred_fake_cloth_2 = D_CB(tcloth_recv_2)
            
            loss_D_CB_real = criterion_GAN(pred_real_cloth, torch.ones_like(pred_real_cloth))
            loss_D_CB_fake = criterion_GAN(pred_fake_cloth, torch.zeros_like(pred_fake_cloth))
            loss_D_CB_fake_2 = criterion_GAN(pred_fake_cloth, torch.zeros_like(pred_fake_cloth))
            loss_D_CB = 0.5 * (loss_D_CB_real + loss_D_CB_fake) + 0.5 * (loss_D_CB_real + loss_D_CB_fake_2)

            loss_D_CB.backward(retain_graph=True)
            optimizer_D_CB.step()
            
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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    EPOCH,
                    batch_idx,
                    len(train_loader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            #if batches_done % SAMPLE_INTERVAL == 0:
                #sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_CB.step()
        print("sucess")
        