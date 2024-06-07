""" fusion dataset consisting of image and associated audio
"""

import torch
import random
from torch.utils.data import Dataset
from utils.datasets import letterbox
import numpy as np
from utils.general import check_img_size
from Data_Augmentation import  add_gaussian_noise, add_saltpepper_noise
import cv2
from augment import SpecAugment

import math


def transform_img(image):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # create the translation matrix using tx and ty, it is a NumPy array 
        tx, ty = w * random.uniform(-0.1,0.1), h * random.uniform(-0.1,0.1)
        M = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        translated = cv2.warpAffine(image, M, (w, h))

        # rotate our image by 45 degrees around the center of the image
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(translated, M, (w, h))

        return rotated

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
            

class FusionDataset(Dataset):

    def __init__(self, data_id, img, audio, labels, img_size):
        self.data_id, self.audio, self.labels = data_id, audio, labels
        self.img = img
        if data_id == 1:
            self.augment = True
        else:
            self.augment = False 

        if data_id == 4:
            self.orig = True
        else: 
             self.orig = False
        if data_id == 3:
             self.test = True
        else: 
             self.test = False

        self.size = img_size   
       
    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        label = self.labels[index] #torch.size: [1]
        image_path = "data/" + self.img[index]
        #print(image_path)
        audio = self.audio[index] 
        #img = self.img[index]
        #print(image_path)
        img0 = cv2.imread(image_path)  # BGR
        img = img0.copy()
        #cv2.imwrite('D:/Masterarbeit/ausgangsbild.jpg', img0) 
        #print(img0.shape)
        # Padded resize
        img = letterbox(img, (self.size, self.size))[0]
        #cv2.imwrite('D:/Masterarbeit/output.jpg', img) 
        print(img.shape)
        #img = cv2.resize(img, (640, 480))
        img = cv2.resize(img, (1280, 736))
        #cv2.imshow('Ausgangsbild',img0) 
        #cv2.imshow('Nach Resize:', img) 
 
        #cv2.waitKey(0) 
        if self.augment:
        #    if random.uniform(0,1) < 0.5:
        #        img = cv2.flip(img, 1)
            
        #    img = transform_img(img)
            augment_hsv(img)
        #    if random.uniform(0,1) < 0.5:
        #        audio = add_gaussian_noise(audio)
        #    if random.uniform(0,1) < 0.5:
        #        audio = add_saltpepper_noise(audio)
        #    if random.uniform(0,1) < 0.5: 
        #       apply = SpecAugment(audio, 'max')
        #       audio = apply.time_mask()
            if random.uniform(0,1) < 0.5: 
                apply = SpecAugment(audio, 'max')
                audio = apply.freq_mask()  
        # closing all open windows 
        #cv2.destroyAllWindows() 
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if self.orig:
             return image_path, img0, img, label
        if self.test:
             return img, audio, label, image_path
        else: 
            return img, audio, label
