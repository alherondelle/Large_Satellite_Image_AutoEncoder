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
          for files_ in list_of_file:
              self.data.append(os.path.join(folder,files_))
        self.path = path_

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
        img_ = self.data[index]
        image = np.load(os.path.join(self.path,img_)) - 0.2158
        image = torch.from_numpy(image.astype(np.float64))

        return image, img_



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
        self.conv1 = nn.Conv2d(2, 16, 3, padding = 1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        print('conv1: ',x.shape)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        print('conv2: ',x.shape)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        print('conv3: ',x.shape)
                
        return x


# Model initialization and weights loading
ae = Autoencoder().cuda()
#ae = nn.Sequential(*list(ae.children()))
if opt.start_epoch != 0:
  ae.load_state_dict(torch.load("./conv_encoder_image_v2_%d.pth" % (opt.start_epoch)))

ae.eval()

# Dataset info for metrics computing 

iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

for i, (img, image_name) in tqdm(enumerate(data_loader)):
    print(torch.max(img), 'is max')
    print(torch.min(img), 'is min')
    img_ = Variable(img[:,:,:608, :608]).cuda()
        # ===================forward=====================
    output = ae(img_.float())
        # ===================backward====================
    pic = np.array(output[0].cpu().detach())
    # ===================log========================
    image_name = str(image_name[0][:-4])
    month_info = image_name.split('/')[0]
    #if not os.path.exists('./image_model_encoding_test/'+month_info):
    #    os.mkdir('./image_model_encoding_test/'+month_info)
    #np.save('./image_model_encoding_test/'+image_name+'.npy', pic)


