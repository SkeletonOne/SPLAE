import os
import numpy as np
import cv2
import SimpleITK as sitk
import torch, torchvision
import scipy.ndimage
import random
from PIL import Image

#Define some data augmentation functions here. Can be called in main.py

def select_data_for_augmentation(img_path, count):
    for dirName, subdirList, fileList in os.walk(img_path):
            print('before data augmentation, we have: ',count, 'images for training.')
            fileList = sorted(fileList)
            for filename in fileList:
                if "mask" not in filename.lower():
                  img, lbl = Image.open(img_path+filename), Image.open(img_path+filename[:-4]+'_mask.png')
                  # Do rotate
                  img_rot, lbl_rot = rotate(img, lbl)
                  cv2.imwrite('./imgs/'+str(count)+'.png',np.array(img_rot))
                  cv2.imwrite('./imgs/'+str(count)+'_mask.png',np.array(lbl_rot))
                  count+=1
                  # Do horizontal flip
                  img_hf, lbl_hf = horizontal_flip(img, lbl)
                  cv2.imwrite('./imgs/'+str(count)+'.png',np.array(img_hf))
                  cv2.imwrite('./imgs/'+str(count)+'_mask.png',np.array(lbl_hf))
                  count+=1
                  # Do vertical flip
                  img_vf, lbl_vf = horizontal_flip(img, lbl)
                  cv2.imwrite('./imgs/'+str(count)+'.png',np.array(img_vf))
                  cv2.imwrite('./imgs/'+str(count)+'_mask.png',np.array(lbl_vf))
                  count+=1
    return count
                
def rotate(img, lbl, theta = None):
    # Rotate volume by a minor angle (+/- 10 degrees: determined by investigation of dataset variability)
    if theta is None:
        theta = random.randint(-10, 10)
    img_rot = scipy.ndimage.interpolation.rotate(img, theta, reshape = False)
    lbl_rot = scipy.ndimage.interpolation.rotate(lbl, theta, reshape = False)
    return img_rot, lbl_rot

def horizontal_flip(img, lbl):
    img_hf = torchvision.transforms.functional.hflip(img)
    lbl_hf = torchvision.transforms.functional.hflip(lbl)
    return img_hf, lbl_hf

def vertical_flip(img, lbl):
    img_vf = torchvision.transforms.functional.vflip(img)
    lbl_vf = torchvision.transforms.functional.vflip(lbl)
    return img_vf, lbl_vf