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
import sys
sys.path.append("../distance_estimation")
import image_warp
# from facial_recog_new import predict
# from facial_recog_new import lib
# import /home/firmware/crc_fa22/unpushed/Neural-Networks-Self-Driving-Car-Raspberry-Pi-main/Step1-Data-Collection/predict

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,10)

counter = 0
output = "None"

def getImg(display= False,size=[400,240]):
    global counter, output
    counter += 1

    # t = transforms.Compose([transforms.ToPILImage(),
 	#     transforms.Resize((size[0],size[1])),transforms.ToTensor()])

    _, img = cap.read()
    # img = t(img)

    # # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img)

    # # reminder that cnn = convolutional neural network
    # cnn = torch.load("/home/firmware/crc_fa22/unpushed/Neural-Networks-Self-Driving-Car-Raspberry-Pi-main/Step1-Data-Collection/facial_recog_new/face_recog.pth")
    # cnn.eval()

    # output = cnn(img)

    if counter == 10:
        try:
            arr = image_warp.warp(img)
            new_img = image_warp.crop(arr[0], arr[1]) 
            output = predict.func(new_img)
            counter = 0
        except:
            output = "None"
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
