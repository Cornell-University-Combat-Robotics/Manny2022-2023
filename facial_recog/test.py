# copy and pasted from predict.py file
import torch
from lib.core import config
from lib.dataset import dataset
from imutils import paths
import cv2
from torchvision import transforms
import numpy as np

b_train_img_paths = list(paths.list_images(config.B_TRAIN_IMG_PATH))
s_train_img_paths = list(paths.list_images(config.S_TRAIN_IMG_PATH))
# print(b_train_img_paths)

train_image_lst = []
train_label_lst = []

name2num = {0:"Blaze", 1:"Sebastian"}
num_tests = 0
num_correct = 0

transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

# copy and pasted from predict.py file
def predict(file_path):
  img = cv2.imread(file_path)
  img = transforms(img)

  img = np.expand_dims(img, 0)
  img = torch.from_numpy(img)

  cnn = torch.load("/Users/richmjin/Desktop/facial_recog/output/face_recog.pth")
  cnn.eval()

  output = cnn(img)
  predicted_person = torch.argmax(output, dim=1)

  return name2num[predicted_person.numpy()[0]]

for i, e in enumerate(b_train_img_paths):
  if 401<=i<=800:
    num_tests += 1
    if (predict(e) == "Blaze"):
      num_correct += 1

for i, e in enumerate(s_train_img_paths):
  if 401<=i<=800:
    num_tests += 1
    if (predict(e) == "Sebastian"):
      num_correct += 1

print("Accuracy: " + str(num_correct/num_tests))









