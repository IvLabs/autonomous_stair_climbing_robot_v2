import cv2 as cv
import numpy as np
import argparse
import os
import json


def disp_mask(image = None, Mask = None):

    alpha = 0.5
    try:
        image = args.image
        mask = args.mask
        print('--------------------------------')

    except:
        pass

    image = cv.imread(image)
    mask = np.load(mask)

    if image is None:
        print("Error loading src1")
        exit(-1)
    elif mask is None:
        print("Error loading src2")
        exit(-1)
    mask[:,:,1] = 0;
    seg = cv.addWeighted(image, alpha, mask, (1-alpha), 0.0)
    cv.imshow('Segmented', seg)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_mask():

    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
        print('Created out_dir')

    for name in os.listdir(args.input_image):
        frame = cv.imread(args.input_image + '/' + name)
        with open(str('./' + args.input_annotations + '/' + name.strip('.jpg') + '.json')) as json_file:
            data = json.load(json_file)
        key_points = data['shapes'][0]['points']
        mask = np.zeros(frame.shape, np.uint8)
        cv.fillPoly(mask, np.int32([np.array(key_points)]), color = (255,255,255))
        # cv.imshow('temp',mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite(args.directory + '/' + name, mask)


if __name__ == "__get_mask__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    parser.add_argument("-ia", "--input_annotations", type=str, help="Bag file to read")
    parser.add_argument("-ii", "--input_image", type=str, help="Bag file to read")
    args = parser.parse_args()
    main()

if __name__ == "__disp_mask__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path/image")
    parser.add_argument("-m", "--mask", type=str, help="path/mask")
    parser.add_argument("-a", "--alpha", type=int, help="alpha")
    args = parser.parse_args()
    main()
