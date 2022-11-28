import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
# --- Training dataset --- #
import torch as th
import cv2
class ValData(data.Dataset):
    def __init__(self, data_dir, crop_size=[256,256]):
        super().__init__()
        self.train_data_dir = data_dir
      
        input_names=os.listdir(self.train_data_dir)
        self.input_names=input_names[:20]
        self.crop_size = crop_size
        # print(self.input_names)
        self.resolution=128

    def get_images(self, index):
        input_name = self.input_names[index]
 
        thermal_image = self.process_and_load_images(os.path.join(self.train_data_dir  ,input_name))

        out_dict={'thermal': thermal_image, 'Index': input_name}

        return  out_dict

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

    def process_and_load_images(self,path):
        pil_image = Image.open(path)
        pil_image=pil_image.resize((self.resolution,self.resolution))
        arr=np.array(pil_image).astype(np.float32)
        arr=arr/127.5-1.0
        arr = np.transpose(arr, [2, 0, 1])

        return arr
