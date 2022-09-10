# local env: cv
# to add more command line args, see https://docs.python.org/3/howto/argparse.html

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from lib.dataset.dataset import MNIST
from lib.core import config
from lib.model import model
import os
from tqdm import tqdm
import time

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.Compose([transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
      transforms.ToTensor()])

# Load datasets for training and testing.
mnist_training = MNIST(root='/Users/richmjin/Desktop/Projects/digits_recog/mnist', train=True, download=True,transform=t)
mnist_val = MNIST(root='/Users/richmjin/Desktop/Projects/digits_recog/mnist', train=False, download=True, transform=t)


# image, label = mnist_training[0]          # returns PIL image with its labels
# image = image.squeeze(0).numpy()
# image = np.uint8(image*255)
# print(label)
# cv2.imwrite('/home/richard/digits_recog/mnist/sample.jpg', image)  # we get a 1x28x28 tensor -> remove first dimension

# Create a simple neural network with one hidden layer with 256 neurons.
model = model.MODEL.to(config.DEVICE)
model.train()

# Use Adam as optimizer.
opt = torch.optim.Adam(params=model.parameters(), lr=config.INIT_LR)

# Use CrossEntropyLoss for as loss function.
loss_fn = torch.nn.CrossEntropyLoss()

# We train the model with batches of 500 examples.
train_loader = torch.utils.data.DataLoader(mnist_training, 
                  batch_size=config.BATCH_SIZE, shuffle=True, 
                  pin_memory=config.PIN_MEMORY, num_workers=0)  # can be faster by setting num_workers=os.cpu_count()

# Training of the model. We use 10 epochs.
train_losses = []

startTime = time.time()

for epoch in range(config.NUM_EPOCHS):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        # Reshape data from [500, 1, 28, 28] to [500, 784] and use the model to make predictions.
        predictions = model(imgs.view(len(imgs), -1))  
        # Compute the loss.
        loss = loss_fn(predictions, labels)

        # #Replaces pow(2.0) with abs() for L1 regularization
        # l2_lambda, l2_norm = 0.001, sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm

        # explanation: 
        # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_losses.append(float(loss))
    print(f"Epoch: {epoch}, Loss: {float(loss)}")

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(train_losses, label="train_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
# serialize the model to disk
torch.save(model, config.MODEL_PATH)
