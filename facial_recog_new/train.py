import torch
from lib.core import config
from lib.dataset import dataset
from imutils import paths
import cv2
import re
from torchvision import transforms
from torch.utils.data import DataLoader
from lib.model import model
import time
import matplotlib.pyplot as plt

#print("here")
# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

b_train_img_paths = list(paths.list_images(config.B_TRAIN_IMG_PATH))
s_train_img_paths = list(paths.list_images(config.S_TRAIN_IMG_PATH))

train_image_lst = []
train_label_lst = []

name2num = {"Blaze":0, "Sebastian":1}

for i, e in enumerate(b_train_img_paths):
  if 100<=i<=120:
    train_img = cv2.imread(e)
    train_image_lst.append(train_img)
    # splits the file path using "_" and "." characters
    # Example: "/Users/richmjin/Desktop/facial_recog/lib/dataset/input_img/1_Blaze.jpg"
    # --> [... img/1, Blaze, jpg]
    path_segmented = re.split(r"[_.]",e)
    # gets the second to last value in path_segmented (the person's name)
    # add the person's name to our list of labels 
    train_label_lst.append(name2num[path_segmented[-2]])

for i, e in enumerate(s_train_img_paths):
  if 100<=i<=120:
    train_img = cv2.imread(e)
    train_image_lst.append(train_img)
    path_segmented = re.split(r"[_.]",e)
    train_label_lst.append(name2num[path_segmented[-2]])

# for i, e in enumerate(r_train_img_paths):
#   if 0<=i<=3:
#     train_img = cv2.imread(e)
#     train_image_lst.append(train_img)
#     path_segmented = re.split(r"[_.]",e)
#     train_label_lst.append(name2num[path_segmented[-2]])

# print(train_label_lst)

train_DS = dataset.FRDataset(image = train_image_lst, label = train_label_lst, transforms = transforms)

train_loader = DataLoader(train_DS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)

cnn = model.CNN()
cnn.train()

opt = torch.optim.Adam(params=cnn.parameters(), lr=config.INIT_LR)
loss_fn = torch.nn.NLLLoss()

train_losses = []
#print("here1")
startTime = time.time()

for epoch in range(config.NUM_EPOCHS):
    # print("here1.5")
    for i, (imgs, labels) in enumerate(train_loader):
        #print("here1.75")
        # imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        # Reshape data from [2, 3, 256, 256] to [2, 196608] and use the model to make predictions.
        predictions = cnn(imgs)  

        # Compute the loss.
        #print("here2")
        loss = loss_fn(predictions, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()      
        #print("here3")
        train_losses.append(float(loss))
    print(f"Epoch: {epoch}, Loss: {float(loss)}")

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
# # print("here4")
# # plot the training loss
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(train_losses, label="train_loss")
# plt.title("Training Loss on Dataset")
# plt.xlabel("Batch #")
# plt.ylabel("Loss")
# plt.legend(loc="lower left")
# plt.savefig(config.PLOT_PATH)
# # serialize the model to disk

torch.save(cnn, config.MODEL_PATH)
