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

# Image difference data loader
# Shuffling dataset is not supported at the moment

class METEOSATDataset(Dataset):

    def __init__(self, path_):
        self.data=[]
        for folder in os.listdir(path_):
          list_of_file = os.listdir(os.path.join(path_, folder))
          list_of_file.sort()
          list_img = []
          for file_ in list_of_file:
            list_img.append(os.path.join(folder, file_))
          self.data.append(list_img)
        self.path = path_
        #self.path_init = self.data[0][:6]
        #self.previous_image = np.load(os.path.join(self.path,self.data[0])).astype(np.float64)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_seq = self.data[index]
        idx2 = int(np.trunc((len(video_seq)-1)*random.random()))
        image_1 = np.load(os.path.join(self.path,video_seq[idx2]))
        image_1 = torch.from_numpy(image_1.astype(np.float64))
        image_2 = np.load(os.path.join(self.path,video_seq[idx2+1]))
        image_2 = torch.from_numpy(image_2.astype(np.float64))
        img = (image_2 - image_1 + 2)/4
        return img


# Training parameters 
num_epochs = 200
batch_size = 64
learning_rate = 1e-3
train_img = '../SatellitePredictionGAN/data/METEOSAT/train'
start_epoch = 0
end_epoch = 150

# Data loader 
data_loader = torch.utils.data.DataLoader(dataset=METEOSATDataset(train_img),
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=0)

# Model architecture 

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 2, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))
                
        return x


# Model initialization and weights loading

ae = Autoencoder().cuda()
if start_epoch != 0:
  ae.load_state_dict(torch.load("AE_%d.pth" %  start_epoch))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, weight_decay=1e-5)

# Dataset info for metrics computing 

iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# Training

for epoch in range(start_epoch, end_epoch):
    t0 = time()
    for i, img in tqdm(enumerate(data_loader)):
      img_ = Variable(img[:,:,:1420, :604]).cuda()
        # ===================forward=====================
      output = ae(img_.float())
      loss = criterion(output, img_.float())
        # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, time:{:.4f}'
          .format(epoch+1, end_epoch, loss.item()*100, time() - t0))
    if epoch % 10 == 0:
        torch.save(ae.state_dict(), './conv_autoencoder_{}.pth'.format(epoch))
        pic = output[0].cpu()
        real_pic = img_[0].cpu()
        save_image(pic, './image_{}.png'.format(epoch))
        save_image(real_pic, './image_real_{}.png'.format(epoch))

# Saving trained model : Final
torch.save(ae.state_dict(), './conv_autoencoder_{}.pth'.format(epoch))

# Stopping train phase 
#ae.eval()

# Save the trained model once the training is over: 
torch.save(ae.state_dict(),  "AE_%d.pth" % (epoch))

