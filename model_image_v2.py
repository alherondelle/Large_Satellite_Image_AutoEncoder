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
        image = np.load(os.path.join(self.path,img_))
        image = torch.from_numpy(image.astype(np.float64))
        return image



# Data loader 
data_loader = torch.utils.data.DataLoader(dataset=METEOSATDataset(opt.train_img),
                                            batch_size=opt.batch_size,
                                            shuffle=True, num_workers=0)

# Model architecture 

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        #encoder
        in_features = 604*604
        out_features = in_features // 4
        self.enc1 = nn.Linear(in_features=in_features, out_features=out_features)
        in_features = out_features
        out_features = in_features//2
        self.enc2 = nn.Linear(n_features=in_features, out_features=out_features)
        in_features = out_features
        out_features = in_features//2
        self.enc3 = nn.Linear(n_features=in_features, out_features=out_features)
        in_features = out_features
        out_features = in_features//2
        self.enc4 = nn.Linear(n_features=in_features, out_features=out_features)
        in_features = out_features
        out_features = in_features//2
        self.enc5 = nn.Linear(in_features=in_features, out_features=out_features)

        # decoder 
        in_features = out_features
        out_features =in_features *2
        self.dec1 = nn.Linear(in_features=in_features, out_features=out_features)
        in_features = out_features
        out_features =in_features *2
        self.dec2 = nn.Linear(in_features=in_features, out_features=out_features)
        in_features = out_features
        out_features =in_features *2
        self.dec3 = nn.Linear(in_features=in_features, out_features=out_features)
        in_features = out_features
        out_features =in_features *2
        self.dec4 = nn.Linear(in_features=in_features, out_features=out_features)
        in_features = out_features
        out_features =in_features *4
        self.dec5 = nn.Linear(in_features=in_features, out_features=out_features)


    def forward(self, x):
        #encoder
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        print(x.shape)
        #decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
                
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
      img_ = Variable(img[:,:,:604, :604].reshape(opt.batch_size, 2, 604*604)).cuda()
        # ===================forward=====================
      output = ae(img_.float()).reshape(opt.batch_size, 2, 604, 604)
      print(output.shape) 
      loss = criterion(output, img_.float())
        # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, time:{:.4f}'
          .format(epoch+1, opt.end_epoch, loss.item()*100, time() - t0))
    if epoch % 10 == 0:
        torch.save(ae.state_dict(), './conv_autoencoder_model_v2_{}.pth'.format(epoch))
        pic = output[0].cpu().detach()
        real_pic = img_[0].cpu().detach()
        save_image(pic, './image_model_v2_{}.png'.format(epoch))
        save_image(real_pic, './image_real_model_v2_{}.png'.format(epoch))

# Saving trained model : Final
torch.save(ae.state_dict(), './conv_autoencoder_model_v2_{}.pth'.format(epoch))

"""# Stopping train phase & Separating encoder / decoder 
list_ae = list(ae.children())

ae_encoder = nn.Sequential(*list_ae[:-2]).cuda()
ae_decoder = nn.Sequential(*list_ae[-2:]).cuda()
ae_encoder.eval()
ae_decoder.eval()

torch.save(ae_encoder.state_dict(), "./conv_encoder_image_%d.pth" % (opt.start_epoch))
torch.save(ae_decoder.state_dict(), "./conv_decoder_image_%d.pth" % (opt.start_epoch))

# Save the trained model once the training is over: 
torch.save(ae.state_dict(),  "./conv_autoencoder_model_{}.pth".format(epoch))
"""