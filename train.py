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
from model import *
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


#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=MOMENTUM)
#device = torch.device(GPU_ID)


#torch.cuda.memory_summary(device = GPU_ID, abbreviated=False)

inp_cloth_enc = nn.DataParallel(ResidualBlock(in_features = 3)).cuda()
inp_body_enc = nn.DataParallel(ResidualBlock(in_features = 3)).cuda()
G_CB = nn.DataParallel(GeneratorResNet(input_shape = (2*TRAIN_BATCH_SIZE, 3,256,256))).cuda()
G_BC = nn.DataParallel(GeneratorResNet(input_shape = (2*TRAIN_BATCH_SIZE, 3,256,256))).cuda()
D_CB = nn.DataParallel(Discriminator(input_shape = (2*TRAIN_BATCH_SIZE, 3,256,256))).cuda()
D_BC = nn.DataParallel(Discriminator(input_shape = (2*TRAIN_BATCH_SIZE, 3,256,256))).cuda()

#Buffers of previously generated samples
fake_CB_buffer = ReplayBuffer()
fake_BC_buffer = ReplayBuffer()


#x = torch.randn(3, 3, 224, 224).to(GPU_ID)
#output = model(x).to(GPU_ID)
#print(output.size())
#summary(model, (3, 224, 224))

gc.collect()
torch.cuda.empty_cache() 

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#Optimizers
optimizer_G = optimizer = optim.SGD(filter(lambda p: p.requires_grad, G_CB.parameters()), lr=LR, momentum=MOMENTUM)
optimizer_D_CB = torch.optim.Adam(D_CB.parameters(), lr=LR) 
optimizer_D_BC = torch.optim.Adam(D_BC.parameters(), lr=LR)

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(EPOCH, 0,  EPOCH*0.5).step
)
lr_scheduler_D_CB = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_CB, lr_lambda=LambdaLR(EPOCH, 0, EPOCH*0.5).step
)
lr_scheduler_D_BC = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_BC, lr_lambda=LambdaLR(EPOCH, 0, EPOCH*0.5).step
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(test_loader))
    G_CB.eval()
    G_BC.eval()
    target_cloth = torch.tensor(imgs[0]).cuda()
    target_body = torch.tensor(imgs[1]).cuda()
    G_CB.eval()
    cloth = inp_cloth_enc.forward(target_cloth)
    body = inp_body_enc.forward(target_body)

    cloth_body = torch.cat([cloth, body], dim=0)
    body_cloth = torch.cat([body, cloth], dim=0)
            
    fake_CB = G_CB(cloth_body)
    fake_BC = G_BC(body_cloth)
    # Arange images along x-axis
    real_A = make_grid(target_cloth, nrow=5, normalize=True)
    real_B = make_grid(target_body, nrow=5, normalize=True)
    fake_A = make_grid(fake_CB, nrow=5, normalize=True)
    fake_B = make_grid(fake_BC, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (DATASET_NAME, batches_done), normalize=False)
'''target_body.shape:  torch.Size([4, 3, 256, 256])
cloth.shape:  torch.Size([4, 3, 256, 256])
body.shape:  torch.Size([4, 3, 256, 256])
cloth_body.shape:  torch.Size([8, 3, 256, 256])
fake_CB[:4,:,:,:].shape:  torch.Size([4, 3, 256, 256])
output shape:  torch.Size([8, 3, 256, 256])
D_CB(cloth_body).shape:  torch.Size([4, 1, 16, 16])
valid.shape:  torch.Size([4, 1, 16, 16])
'''


if __name__ == '__main__':
    G_CB.train()
    G_BC.train()
    inp_cloth_enc.train()
    inp_body_enc.train()
    
    prev_time = time.time()
    for epoch in range(1, EPOCH +1):
        for (batch_idx, target) in enumerate(train_loader): 
            #print("target.size: ",target.size)
            save_inp = "results/210726/inputs"
            save_out = "results/210726/outputs"
            target_cloth = torch.tensor(target[0], dtype=torch.float32)
            save_inp = save_inp+str(batch_idx)+".jpg"
            save_image(target_cloth, save_inp)
            target_body = torch.tensor(target[1], dtype=torch.float32)
            #print('target_cloth.shape: ',target_cloth.shape)
            #print('target_body.shape: ',target_body.shape)
            #G_CB.train()
            #inp_cloth_enc.train()
            #inp_body_enc.train()
            cloth = inp_cloth_enc.forward(target_cloth)
            body = inp_body_enc.forward(target_body)
            #print('cloth.shape: ',cloth.shape)
            #print('body.shape: ',body.shape)
            
            # Adversarial ground truths
            valid = torch.tensor(np.ones((target_cloth.size(0), *D_CB.module.output_shape)), requires_grad=False, dtype=torch.float32).cuda()
            fake = torch.tensor(np.zeros((target_body.size(0), *D_BC.module.output_shape)), requires_grad=False, dtype=torch.float32).cuda()

            
            cloth_body = torch.cat([cloth, body], dim=0)
            body_cloth = torch.cat([body, cloth], dim=0)
            #print('cloth_body.shape: ',cloth_body.shape)
            fake_CB = G_CB.forward(cloth_body)
            fake_BC = G_BC.forward(body_cloth)
            optimizer_G.zero_grad()
            
            # Identity loss
            #print("fake_CB[:4,:,:,:].shape: ",fake_CB[:4,:,:,:].shape)
            loss_id_A = criterion_identity(fake_CB, cloth_body)
            loss_id_B = criterion_identity(fake_BC, body_cloth)
            loss_identity = (loss_id_A + loss_id_B) / 2
            
            # GAN loss
            loss_GAN_AB = criterion_GAN(fake_CB, cloth_body) #MSE
            loss_GAN_BA = criterion_GAN(fake_BC, body_cloth)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_CB = G_CB(fake_CB)
            loss_cycle_A = criterion_cycle(recov_CB, cloth_body)
            recov_BC = G_BC(fake_BC)
            loss_cycle_B = criterion_cycle(recov_BC, body_cloth)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + LAMBDA_CYC * loss_cycle + LAMBDA_ID * loss_identity
            
            #out  = outputs.numpy()
            #out = outputs[0].detach().cpu().clone().numpy().transpose(0,1,2)
            #print('output shape: ',fake_CB.shape)
            save_out = save_out+str(batch_idx)+".jpg"
            save_image(fake_CB, save_out)

            loss_G.backward(retain_graph=True)
            optimizer_G.step()
            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_CB.zero_grad()

            # Real loss
            #print("D_CB(cloth_body).shape: ",D_CB(cloth_body)[:4,:,:,:].shape) # torch.Size([8, 1, 16, 16])
            #print("valid.shape: ",valid.shape)
            loss_real = criterion_GAN(D_CB(cloth_body)[:4,:,:,:], valid)
            # Fake loss (on batch of previously generated samples)
            fake_CB_ = fake_CB_buffer.push_and_pop(fake_CB)
            loss_fake = criterion_GAN(D_CB(fake_CB_.detach())[:4,:,:,:], fake)
            # Total loss
            loss_D_CB = (loss_real + loss_fake) / 2

            loss_D_CB.backward(retain_graph=True)
            optimizer_D_CB.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_BC.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_BC(body_cloth)[4:,:,:,:], valid)
            # Fake loss (on batch of previously generated samples)
            fake_BC_ = fake_BC_buffer.push_and_pop(fake_BC)
            loss_fake = criterion_GAN(D_BC(fake_BC_.detach())[4:,:,:,:], fake)
            # Total loss
            loss_D_BC = (loss_real + loss_fake) / 2

            loss_D_BC.backward(retain_graph=True)
            optimizer_D_BC.step()

            loss_D = (loss_D_CB + loss_D_BC) / 2

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
        lr_scheduler_D_BC.step()

        print("sucess")
        