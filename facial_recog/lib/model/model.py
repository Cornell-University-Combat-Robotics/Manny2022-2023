import torch
import torch.nn.functional as F

# CNN = Convolutional Neural Network
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # the 3 args are color_in_chan, out_chan, kernel_size
        # we can choose a num for out_chan
        # reference: https://www.youtube.com/watch?v=LgFNRIFxuUo
        # sidenote: back and white images would use a color_in_chan of 1
        self.conv1 = torch.nn.Conv2d(3,10,kernel_size=5) 

        # the in_chan here needs to mathc the out_chan of the previous line
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        
        self.max_pool = torch.nn.MaxPool2d(2)
        
        # the first argument needs to start out as a placeholder
        # later adjust this value according to error messages
        # "creates a single layer feed-forward network with n inputs and m outputs"
        self.flatten_conv = torch.nn.Linear(74420,2) 
        
    def forward(self,x):
        in_size = x.size(0) # in_size is the size of one batch
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = x.view(in_size,-1)
        x = self.flatten_conv(x)
        return F.log_softmax(x,dim=1)