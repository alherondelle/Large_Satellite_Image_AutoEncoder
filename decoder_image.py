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
parser.add_argument('--train_img', default='../SatellitePredictionGAN/saved_vectors/METEOSAT/test', type=str, help ='Path to training dataset')
opt = parser.parse_args()

# Image difference data loader
class METEOSATDataset(Dataset):

    def __init__(self, path_):
        self.data=[]
        for file_ in os.listdir(path_):
            self.data.append(file_)
        """for folder in os.listdir(path_):
          list_of_file = os.listdir(os.path.join(path_, folder))
          for files_ in list_of_file:
              self.data.append(os.path.join(folder,files_))"""
        self.path = path_

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
        img_ = self.data[index]
        image = torch.load(os.path.join(self.path,img_))
        print(image[0].shape)
        return image[0], img_



# Data loader 
data_loader = torch.utils.data.DataLoader(dataset=METEOSATDataset(opt.train_img),
                                            batch_size=1,
                                            shuffle=True, num_workers=0)

# Model architecture 

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 2, 2, stride=2)

    def forward(self, x):
        ## add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv3(x))
                
        return x


# Model initialization and weights loading
ae = Autoencoder().cuda()
#ae = nn.Sequential(*list(ae.children()))
if opt.start_epoch != 0:
  ae.load_state_dict(torch.load("./conv_decoder_image_v2_%d.pth" % (opt.start_epoch)))

ae.eval()

# Dataset info for metrics computing 

iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

for i, (img, image_name) in tqdm(enumerate(data_loader)):
    img_ = Variable(img).cuda()
        # ===================forward=====================
    output = ae(img_.float())
        # ===================backward====================
    print('stop')
    pic = np.array(output[0].cpu().detach())
    # ===================log========================
    #image_name = str(image_name[0][:-4])
    print(str(image_name))
    #month_info = image_name.split('/')[0]
    """if not os.path.exists('./image_model_decoding_test/'+month_info):
        os.mkdir('./image_model_decoding_test/'+month_info)"""
    np.save('./image_model_decoding_test/'+str(image_name), pic)