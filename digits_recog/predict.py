# to add more command line args, see https://docs.python.org/3/howto/argparse.html

import torch
from lib.dataset.dataset import MNIST
from torchvision import datasets, transforms
from lib.core import config
import time

startTime = time.time()

# Determine the accuracy of our clasifier
# =======================================
t = transforms.Compose([transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
      transforms.ToTensor()])
mnist_val = MNIST(root='/Users/richmjin/Desktop/Projects/digits_recog/mnist', train=False, download=True, transform=t)

# Load all 10000 images from the validation set.
n = 10000
loader = torch.utils.data.DataLoader(mnist_val, batch_size=n)
images, labels = iter(loader).next()
images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

# The tensor images has the shape [10000, 1, 28, 28]. Reshape the tensor to
# [10000, 784] as our model expected a flat vector.
data = images.view(n, -1)

model = torch.load(config.MODEL_PATH).to(config.DEVICE)
model.eval()

# Use our model to compute the class scores for all images. The result is a
# tensor with shape [10000, 10]. Row i stores the scores for image images[i].
# Column j stores the score for class j.
predictions = model(data)

# For each row determine the column index with the maximum score. This is the
# predicted class.
predicted_classes = torch.argmax(predictions, dim=1)

# Accuracy = number of correctly classified images divided by the total number
# of classified images.
print(sum(predicted_classes.cpu().numpy() == labels.cpu().numpy()) / n)

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))