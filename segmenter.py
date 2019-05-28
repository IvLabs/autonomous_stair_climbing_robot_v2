import torchvision
import numpy as np
import time
from fastai.vision import *

# from sensor_msgs.msg import CompressedImage


#!/usr/bin/env python
# import rospy

import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import os
import cv2 as cv
import torch
from torch.autograd.variable import Variable
from torchvision.transforms import Normalize
import sys
import numpy as np
from PIL import Image

imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# try:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')   # It causes cv2 import error
# except:
#     print('yeah!')

path_mask = '../new_dataset/mask/'
path_img_mask = '../new_dataset/image_mask/'
path_img = '/home/rex/test/new_dataset/image/image/image/'
def preprocess(images):
    images = torch.unsqueeze(torch.from_numpy(images),dim=0)
    images = images.float()
    images = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(images)
    return images#.half()

def get_subtracted(images):
    images = images.view(images.size(0)/2, 2, 3, 480, 640)
    images1, images2 = images[:,0], images[:,1]
    return images1-images2

def get_masked(img, mask):
    masked_image = np.multiply(img, mask/255)
    return masked_image

def getnp(mat):
    mat = mat.cpu()
    return mat.detach().numpy()


if __name__ == '__main__':
    #fastai models
    seg_learner_path = './models/segmentation_model/'
    seg_learn = load_learner(seg_learner_path).to_fp32()
    list = os.listdir(path_img)
    print('Press Ctrl+C for exiting')
    for name in list:
        test_image = open_image(path_img + name)
        img_segment = seg_learn.predict(test_image)[0]
        torchvision.utils.save_image(img_segment.data, path_mask + name)
        masked = np.multiply(test_image.data, img_segment.data)
        torchvision.utils.save_image(masked, path_img_mask + name)
        print('.')
