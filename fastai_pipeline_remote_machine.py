import torchvision
import numpy as np
import time
from fastai.vision import *

from sensor_msgs.msg import CompressedImage


#!/usr/bin/env python
import rospy

import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv
import torch
from torch.autograd.variable import Variable
from torchvision.transforms import Normalize
import sys
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
import numpy as np
from PIL import Image

imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# try:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')   # It causes cv2 import error
# except:
#     print('yeah!')


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

def get_control(msg):
    t1 = time.time()
    frame = cv.imdecode(np.frombuffer(msg.data, dtype=np.uint8), 1)
    cv.imwrite('buffer.jpg', frame)
    test_image = open_image('buffer.jpg')
    img_segment = seg_learn.predict(test_image)[0]

    masked = np.multiply(test_image.data, img_segment.data)
    torchvision.utils.save_image(masked, 'bh_test_masked.jpg')

    masked = open_image('bh_test_masked.jpg')
    action = bh_learn.predict(masked)

    # publish action on a node
    pub = rospy.Publisher('/in_put', Int32 , queue_size=1)
    try:
        print(int(action[0]))
        a = pub.publish(int(action[0]))
        print(a)
    except:
        print('could not publish')
    print(time.time() - t1)



if __name__ == '__main__':
    #fastai models
    seg_learner_path = './models/segmentation_model/'
    bh_learner_path = './models/bh_cloning_model/'
    seg_learn = load_learner(seg_learner_path).to_fp32()
    bh_learn = load_learner(bh_learner_path).to_fp32()

    print('Press Ctrl+C for exiting')
    rospy.init_node('detector', anonymous=True)
    rospy.Subscriber("/output/image_raw/compressed", CompressedImage, get_control)
    rospy.spin()
