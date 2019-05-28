import numpy as np

import os
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv

PATH_IMG = './image/substracted'
PATH_MASK = './mask/substracted'
OUT_FOLDER = './masked_image/sub'

for i in os.listdir(PATH_IMG):
    img, mask = cv.imread(f'{PATH_IMG}/{i}'), cv.imread(f'{PATH_MASK}/{i}')
    masked_image = np.multiply(img, mask/255)
    cv.imwrite(f'{OUT_FOLDER}/{i}', masked_image)
