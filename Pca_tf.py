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
from sklearn.decomposition import PCA

pca_file = open('PCA_components.pkl','rb')
pca_tf = pickle.load(pca_file)

if not os.path.exists('../SatellitePredictionGAN/data/METEOSAT_PCAtf'):
    os.mkdir('METEOSAT_PCAtf')
    os.mkdir('METEOSAT_PCAtf/train')
    os.mkdir('METEOSAT_PCAtf/test')
for folder_ in os.listdir('../SatellitePredictionGAN/data/METEOSAT'):
    for folder_2 in os.listdir(os.path.join('../SatellitePredictionGAN/data/METEOSAT',folder_)):
        if os.path.isdir(os.path.join('../SatellitePredictionGAN/data/METEOSAT',folder_, folder_2)) : 
            for file_ in os.listdir(os.path.join('../SatellitePredictionGAN/data/METEOSAT',folder_, folder_2)):
                img = np.load(os.path.join('../SatellitePredictionGAN/data/METEOSAT', folder_, folder_2, file_))
                x,y,c = img.shape
                img = img/1024.0
                img = img.reshape(x*y, 3)
                img = pca_tf.transform(img).astype(np.float16)
                img = img.reshape(x, y, 2)
                mg = np.moveaxis(img, 2, 0)
                if not os.path.exists(os.path.join('METEOSAT_PCAtf', folder_, folder_2)):
                    os.mkdir(os.path.join('METEOSAT_PCAtf', folder_, folder_2,)
                np.save(os.path.join('METEOSAT_PCAtf', folder_, folder_2, file_), img)

