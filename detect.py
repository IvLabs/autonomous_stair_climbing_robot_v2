#!/usr/bin/env python
import rospy

import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
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

def get_control():
    cap = cv2.VideoCapture(1)
    while(True):
        ret, frame = cap.read()

        print('i am here')
        with torch.no_grad():
            #images = np.random.randn((30, 3, 480, 640)) # get these from a node
            image = preprocess(frame.transpose(2,0,1))
            seg_maps = segmentation_model(image/255)
            masked_image = get_masked(image, seg_maps[:,0,:,:].numpy())
            print(masked_image.shape)
            img = ((masked_image[0,0,:,:].detach().numpy()).astype('uint8'))
            img = Image.fromarray(img)
            img.save('here.jpg')
            #cv2.imwrite('this.jpg', masked_image.numpy())
            #images
            #cv.imwrite(masked_image, 'file.jpg')
            action = bh_model(torch.tensor(masked_image))#.half())
            # action = bh_model(get_subtracted(images))
            #
            # #segmentation maps
            # action = bh_model(seg_maps)
            # action = bh_model(get_subtracted(seg_maps))
            #
            # #seg_maps + images
            # action = bh_model(masked_image(images, seg_maps))
            # action = bh_model(get_subtracted((masked_image(images, seg_maps))))

            _, action = torch.max(torch.nn.Softmax(dim=1)(action),1)
            action = action.numpy()
            print(action[0])

            # publish action on a node
            pub = rospy.Publisher('in_put', Int32 , queue_size=10)
            rate = rospy.Rate(10) # 10hz
            while not rospy.is_shutdown():
                pub.publish(action[0])
                # rate.sleep()


if __name__ == '__main__':
    segmentation_model = torch.load('model/unet_cpu.pth',map_location='cpu')
    segmentation_model = segmentation_model.float()
    bh_model = torch.load('model/masked_img_model_cpu.pth',map_location='cpu')
    bh_model = bh_model.float()
    print('Press Ctrl+C for exiting')
    rospy.init_node('detector', anonymous=True)
    # rospy.Subscriber("img_raw", String, get_control)
    get_control()
    print('reached')
    rospy.spin()
