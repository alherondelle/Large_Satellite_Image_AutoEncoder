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
parser.add_argument('--start_epoch', default=0, type=float, help='starting epoch')
parser.add_argument('--train_img', default='../SatellitePredictionGAN/data/METEOSAT/train', type=str, help ='Path to training dataset')
opt = parser.parse_args()

# Image difference data loader
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
        self.len_seq = len(self.data[0])

    def __len__(self):
      len_ = 0
      for i in self.data:
        len_ += (len(i) -1)
      return len_

    def __getitem__(self, index):
        idx_1 = index // (self.len_seq - 1)
        idx_2 = index % (self.len_seq - 1)
        video_seq = self.data[idx_1]
        image_1 = np.load(os.path.join(self.path,video_seq[idx_2]))
        image_1 = torch.from_numpy(image_1.astype(np.float64))
        image_2 = np.load(os.path.join(self.path,video_seq[idx_2+1]))
        image_2 = torch.from_numpy(image_2.astype(np.float64))
        img = (image_2 - image_1 + 2)/4
        return img, video_seq[idx_2+1]



# Data loader 
data_loader = torch.utils.data.DataLoader(dataset=METEOSATDataset(opt.train_img),
                                            batch_size=1,
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


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
                
        return x


# Model initialization and weights loading
ae = Autoencoder().cuda()
ae = nn.Sequential(*list(ae.children()))
if opt.start_epoch != 0:
  ae.load_state_dict(torch.load("./conv_encoder_%d.pth" % (opt.start_epoch)))

ae.eval()

# Dataset info for metrics computing 

iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# Training
for i, (img, image_name) in tqdm(enumerate(data_loader)):
    img_ = Variable(img[:,:,:1420, :604]).cuda()
        # ===================forward=====================
    output = ae(img_.float())
        # ===================backward====================
    pic = np.array(output[0].cpu().detach())
    # ===================log========================
    image_name = str(image_name[0][:-4])
    month_info = image_name.split('/')[0]
    if not os.path.exists('./image_diff_encoding/'+month_info):
        os.mkdir('./image_diff_encoding/'+month_info)
    print(image_name)
    np.save('./image_diff_encoding/'+image_name+'.npy', pic)