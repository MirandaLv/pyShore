
import os
import torch
import random
import numpy as np
from tqdm import tqdm
# from osgeo import gdal
import rasterio
from os.path import dirname as up
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import pytorch_lightning as pl
import pandas as pd


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# NAIP data
bands_std = np.array([53.4146, 53.4118, 53.4115, 53.4158]).astype('float32')
bands_mean = np.array([98.8948, 98.8902, 98.8835, 98.8795]).astype('float32')

# # Setting dataset folder and its weights
# # ********************************************************************************************

dataset_dir = os.path.join(up(up(up(__file__))), 'dataset_evaluation') #, '256_images_lesstrain'

# # # Pixel-Level class distribution for each dataset
# """
# {'Image_allyear_merged_512': array([0.49019898, 0.44645175, 0.02657669, 0.03677258]),
#  'Image_after_2010_merged_512': array([0.45526231, 0.43276168, 0.05267328, 0.05930273]),
#  'Image_after_2010_merged_256': array([0.32490837, 0.41855632, 0.17178225, 0.08475305]),
#  'Image_allyear_merged_256': array([0.44056167, 0.41559786, 0.09143975, 0.05240071]),
#  'Image_allyear_merged_1024': array([0.50426896, 0.4565353 , 0.01033589, 0.02885985]),
#  'Image_after_2010_merged_1024': array([0.49582348, 0.43486081, 0.02433212, 0.04498359]),
#  'Image_after_2010_VA_512': array([0.47100772, 0.44772889, 0.01990966, 0.06135373]),
#  'Image_after_2010_VA_256': array([0.37283283, 0.48029399, 0.04961892, 0.09725425]),
#  'Image_allyear_VA_512': array([0.49843776, 0.45395527, 0.01021636, 0.03739061]),
#  'Image_allyear_VA_256': array([0.47332474, 0.44650446, 0.02387324, 0.05629757])}
# """


# if data_name == 'Image_after_2010_merged_512':
#     class_distr = torch.Tensor([0.45526231, 0.43276168, 0.05267328, 0.05930273]) # 4 classes
# elif data_name == 'Image_after_2010_merged_256':
#     class_distr = torch.Tensor([0.32490837, 0.41855632, 0.17178225, 0.08475305])
# elif data_name == 'Image_after_2010_merged_1024':
#     class_distr = torch.Tensor([0.49582348, 0.43486081, 0.02433212, 0.04498359])
# elif data_name == 'Image_allyear_merged_256':
#     class_distr = torch.Tensor([0.44056167, 0.41559786, 0.09143975, 0.05240071])
# elif data_name == 'Image_allyear_merged_512':
#     class_distr = torch.Tensor([0.49019898, 0.44645175, 0.02657669, 0.03677258])
# elif data_name == 'Image_allyear_merged_1024':
#     class_distr = torch.Tensor([0.50426896, 0.4565353 , 0.01033589, 0.02885985])
# elif data_name == 'Image_after_2010_VA_512':
#     class_distr = torch.Tensor([0.47100772, 0.44772889, 0.01990966, 0.06135373])
# elif data_name == 'Image_after_2010_VA_256':
#     class_distr = torch.Tensor([0.37283283, 0.48029399, 0.04961892, 0.09725425])
# elif data_name == 'Image_allyear_VA_512':
#     class_distr = torch.Tensor([0.49843776, 0.45395527, 0.01021636, 0.03739061])
# elif data_name == 'Image_allyear_VA_256':
#     class_distr = torch.Tensor([0.47332474, 0.44650446, 0.02387324, 0.05629757])
# else:
#     raise
    
###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################

class ShorelineArmoring(Dataset): # Extend PyTorch's Dataset class

    def __init__(self, mode = 'train', transform=None, standardization=None, data_name = "Image_after_2010_merged_256", path = dataset_dir):
        
        print(path)
              
        if os.path.isdir(os.path.join(path, data_name)):
            
            data_path = os.path.join(path, data_name)
            print(data_path)
        else: 
            raise
        
        if mode=='train':
            self.ROIs = np.array([name for name in os.listdir(os.path.join(data_path, 'train')) if os.path.isfile(os.path.join(data_path, 'train', name))])
                
        elif mode=='test':
            self.ROIs = np.array([name for name in os.listdir(os.path.join(data_path, 'test')) if os.path.isfile(os.path.join(data_path, 'test', name))])
                
        elif mode=='val':
            self.ROIs = np.array([name for name in os.listdir(os.path.join(data_path, 'val')) if os.path.isfile(os.path.join(data_path, 'val', name))])
            
        else:
            raise
        
            
        self.X = []           # Loaded Images
        self.y = []           # Loaded Output masks
            
        for roi in tqdm(self.ROIs, desc = 'Load '+mode+' set to memory'):
            
            # Construct file and folder name from roi
            roi_file = os.path.join(data_path, mode, roi)
            roi_file_mask = os.path.join(data_path, 'masks', roi)
            
            # Load Classsification Mask
            ds = rasterio.open(roi_file_mask)
            temp = np.copy(ds.read().astype(np.int64))

            # Categories from 1 to 0
            temp = np.copy(temp - 1)
            ds=None
            
            self.y.append(temp)
            
            # Load Patch
            ds = rasterio.open(roi_file)
            temp = np.copy(ds.read())
            ds=None
            self.X.append(temp)      

        self.impute_nan = np.tile(bands_mean, (temp.shape[1],temp.shape[2],1))
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.length = len(self.y)
        self.path = data_path
        
    def __len__(self):

        return self.length
    
    def getnames(self):
        return self.ROIs    
    
    def __getitem__(self, index):
        
        img = self.X[index]
        target = self.y[index]

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')       # CHW to HWC
        
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]
        
        if self.transform is not None:
            target = np.moveaxis(target, [0, 1, 2], [2, 0, 1])    # CHW to HWC
            
            stack = np.concatenate([img, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image
        
            stack = self.transform(stack)

            img = stack[:,:,:-1]
            target = stack[:,:,-1].long() # Recast target values back to int64 or torch long dtype
        
        if self.standardization is not None:
            img = self.standardization(img)
        
        return {'image': img, 'mask': target}

        
        



