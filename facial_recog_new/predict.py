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
img = cv2.imread("/Users/richmjin/Desktop/facial_recog/lib/dataset/Blaze1/850_Blaze.png")
img = transforms(img)

# img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)
img = torch.from_numpy(img)

cnn = torch.load("/Users/richmjin/Desktop/facial_recog/output/face_recog.pth")
cnn.eval()

output = cnn(img)
# output = output.squeeze()
predicted_person = torch.argmax(output, dim=1)

name2num = {0:"Blaze", 1:"Sebastian"}

print(output)
print(name2num[predicted_person.numpy()[0]])

# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# for accuracy implementation