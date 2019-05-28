import rospy
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
    # print('yeah!')
from std_msgs.msg import Int32

import sys, select, termios, tty
import os, time, signal, threading
from subprocess import call
from subprocess import Popen
import subprocess
import numpy as np
import cv2
global proc
import csv

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key



if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('stairybot_teleop')
    pub = rospy.Publisher('in_put', Int32 , queue_size=10)

    x = 9
    status = 0
    count = 0

    cap = cv2.VideoCapture(1)
    csv_data = [['FileName','Action']]
    while(True):
        csv_row = []

        key = getKey()
        ret, frame = cap.read()

        if key is not '' or count > 0 :

            if key == ' ' :
                x = 0

            elif key == 'w' :
                x = 5

            elif key == 'd' :
                x = 2

            elif key == 's' :
                x = 4

            elif key == 'a' :
                x = 3

            elif (key == '\x03'):
                break
            pub.publish(x)
            count= count + 1
            print(x)
            imagename = "image" + str(count) + ".jpg"
            cv2.imwrite(imagename,frame)
            csv_row.append(imagename)
            csv_row.append(x)
            csv_data.append(csv_row)


    with open('record.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
        csvFile.close()

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
