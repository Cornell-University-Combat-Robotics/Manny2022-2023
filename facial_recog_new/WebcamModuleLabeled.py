"""
-This module gets an image through the webcam
using the opencv package
-Display can be turned on or off
-Image size can be defined
"""

import torch
from imutils import paths
import cv2
from torchvision import transforms
import numpy as np
import predict
import image_warp
# from facial_recog_new import predict
# from facial_recog_new import lib
# import /home/firmware/crc_fa22/unpushed/Neural-Networks-Self-Driving-Car-Raspberry-Pi-main/Step1-Data-Collection/predict

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,10)

counter = 0
output = "None"
new_img = None
changed = False

def getImg(display= False,size=[500,300]):
    global counter, output
    counter += 1

    _, img = cap.read()
    img = cv2.resize(img,(size[0],size[1]))

    if counter == 10:

        # this is for cropping:
        # ---------------------
        # try:
        #     img = image_warp.warp(img)
        #     img = image_warp.crop(img[0], img[1]) 
        #     img *= 255
        #     img = img.astype(np.uint8)
        #     img = cv2.resize(img,(size[0],size[1]))
        #     output = predict.func(img)
        #     cv2.imwrite("/home/firmware/crc_fa22/unpushed/Neural-Networks-Self-Driving-Car-Raspberry-Pi-main/Step1-Data-Collection/facial_recog_new/test_crop_img/0.jpg",img)
        # except:
        #     output = "None"
        # ---------------------

        # this is for non cropping:
        # ---------------------
        output = predict.func(img)
        # ---------------------

        counter = 0

    img = cv2.putText(img, output, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if display:
        cv2.imshow('IMG',img)
        
    return img

if __name__ == '__main__':
    while True:
        img = getImg(True)


# import cv2

# # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS,10)
# def getImg(display= False,size=[400,240]):
#     _, img = cap.read()
#     img = cv2.resize(img,(size[0],size[1]))
#     img = cv2.putText(img, "text", (230,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     if display:
#         cv2.imshow('IMG',img)
#     return img

# if __name__ == '__main__':
#     while True:
#         img = getImg(True)
