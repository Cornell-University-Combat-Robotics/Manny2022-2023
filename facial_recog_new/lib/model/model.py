import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,10,kernel_size=5) 
        # the 3 args are color_in_chan,out_chan,kernel_size
        # we can choose a num for out_chan
        # reference: https://www.youtube.com/watch?v=LgFNRIFxuUo
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.max_pool = torch.nn.MaxPool2d(2)
        self.flatten_conv = torch.nn.Linear(74420,2) # first arg could be a placeholder
        # Initially do not know input is 320. 
        # Put in a random number, find out based on the error message.
    def forward(self,x):
        in_size = x.size(0) # in_size is the size of one batch
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = x.view(in_size,-1)
        x = self.flatten_conv(x)
        return F.log_softmax(x,dim=1)