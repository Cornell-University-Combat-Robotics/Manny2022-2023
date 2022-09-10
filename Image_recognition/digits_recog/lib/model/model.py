# batch normalization: https://www.youtube.com/watch?v=DtEq44FTPM4&t=383s
# torch.nn.conv2d and torch.nn.maxpool/avgpool: https://www.youtube.com/watch?v=LgFNRIFxuUo

import torch

MODEL = torch.nn.Sequential(
    torch.nn.Linear(28*28, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)


# If a self-implemented Model class is needed, call Model() instead of MODEL
# in train.py.

# The structure is:

# class Model(Module):
# 	def __init__(parameters):
#     ...
#   def forward(self, x):
#     ...

# Ex:
# class CNN(torch.nn.Module):
#     def __init__(self):
#         self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
#         self.max_pool = torch.nn.MaxPool2d(2)
#         self.flatten_conv = torch.nn.Linear(320,10) 
#         # Initially do not know input is 320. 
#         # Put in a random number, find out based on the error message.
#     def forward(self,x):
#         in_size = x.size(0) # in_size is the size of one batch
#         x = F.relu(self.max_pool(self.conv1(x)))
#         x = F.relu(self.max_pool(self.conv2(x)))
#         x = x.view(in_size,-1)
#         x = self.flatten_conv(x)
#         return F.log_softmax(x)