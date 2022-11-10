import torch
from lib.core import config
from lib.dataset import dataset
from imutils import paths
import cv2
from torchvision import transforms
import numpy as np

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

# input image (change path as needed)
img1 = cv2.imread("/Users/bezla/Documents/CRC_Stuff/Manny2022-2023/facial_recog/lib/dataset/Seb1/751_Sebastian.png")
img = transforms(img1)

# img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)
img = torch.from_numpy(img)

cnn = torch.load("/Users/bezla/Documents/CRC_Stuff/Manny2022-2023/facial_recog/output/face_recog.pth")
cnn.eval()

output = cnn(img)
# output = output.squeeze()
predicted_person = torch.argmax(output, dim=1)

name2num = {0:"Blaze", 1:"Sebastian"}

# outputs numbers
# 1st number corresponds to 1st person (Blaze), 2nd number corresponds to 2nd person (Sebastian) and so on
# higher number = greater confidence that the photo is of the associated person
print(output)

# picks out the highest number and prints the associated name of the person
print(name2num[predicted_person.numpy()[0]])

# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# for accuracy implementation
labeled_img = cv2.putText(img1 , name2num[predicted_person.numpy()[0]], (50,50), \
	cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2 )
cv2.imshow('labeled image', labeled_img)
cv2.waitKey(0)