# -*- coding: utf-8 -*-

import glob
import random
import cv2
import os
import torchvision.transforms as transforms
import torch
import random
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np 
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from time import time
from torchvision.utils import save_image
import torch.nn.functional as F
import multiprocessing
import argparse

# Argument parser 

# Training parameters 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--start_epoch', default=0, type=float, help='starting epoch')
parser.add_argument('--end_epoch', default=150, type=float, help='ending epoch')
parser.add_argument('--train_img', default='./METEOSAT_PCAtf/train', type=str, help ='Path to training dataset')
parser.add_argument('--test_img', default='./METEOSAT_PCAtf/test', type=str, help='Path to the testing dataset')
opt = parser.parse_args()

# Image difference data loader
class METEOSATDataset(Dataset):

    def __init__(self, path_):
        self.data=[]
        for folder in os.listdir(path_):
          list_of_file = os.listdir(os.path.join(path_, folder))
          for files_ in list_of_file:
              self.data.append(os.path.join(folder,files_))
        self.path = path_

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
        img_ = self.data[index]
        image = np.load(os.path.join(self.path,img_))
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image)
        return image



# Data loader 
data_loader = torch.utils.data.DataLoader(dataset=METEOSATDataset(opt.train_img),
                                            batch_size=opt.batch_size,
                                            shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=METEOSATDataset(opt.test_img),
                                            batch_size=1,
                                            shuffle=True, num_workers=0)

# Model architecture 

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(2, 16, 3, padding = 1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 2, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)# => compressed representation
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.tanh(self.t_conv3(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        return x


# Model initialization and weights loading
ae = Autoencoder().cuda()
if opt.start_epoch != 0:
  ae.load_state_dict(torch.load("./conv_autoencoder_model_v2_%d.pth" % (opt.start_epoch)))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

# Dataset info for metrics computing 

iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# Training

for epoch in range(opt.start_epoch, opt.end_epoch):
    t0 = time()
    for i, img in tqdm(enumerate(data_loader)):
      img_ = Variable(img[:,:,:608, :608]).cuda()
        # ===================forward=====================
      output = ae(img_.float())
      loss = criterion(output, img_.float())
        # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, time:{:.4f}'
          .format(epoch+1, opt.end_epoch, loss.item()*100, time() - t0))
    if epoch % 5 == 0:
      count = 0
      test_loss = 0
      for i, img in tqdm(enumerate(test_loader)):
        count+= 1
        img_ = Variable(img[:,:,:608, :608]).cuda()
        output = ae(img_.float())
        test_loss += torch.mean((img_.detach().cpu() - output.detach().cpu())**2)

      print('TEST LOSS : ', test_loss.item()/count)

    if epoch % 10 == 0:
        torch.save(ae.state_dict(), './conv_autoencoder_model_v2_{}.pth'.format(epoch))
        pic = output[0].cpu().detach()
        real_pic = img_[0].cpu().detach()
        save_image(pic, './image_model_v2_{}.png'.format(epoch))
        save_image(real_pic, './image_real_model_v2_{}.png'.format(epoch))

# Saving trained model : Final
torch.save(ae.state_dict(), './conv_autoencoder_model_v2_{}.pth'.format(epoch))
# Stopping train phase & Separating encoder / decoder 

ae.eval()
ae_encoder_keys = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", 'pool.weight', 'pool.biais']
ae_decoder_keys = ['t_conv1.weight', 't_conv2.weight', 't_conv3.weight', 't_conv1.bias', 't_conv2.bias', 't_conv3.bias']
ae_encoder_param = {k:v for k,v in ae.state_dict().items() if k in ae_encoder_keys}
ae_decoder_param = {k:v for k,v in ae.state_dict().items() if k in ae_decoder_keys}

torch.save(ae_encoder_param, "./conv_encoder_image_v2_%d.pth" % (epoch))
torch.save(ae_decoder_param, "./conv_decoder_image_v2_%d.pth" % (epoch))

