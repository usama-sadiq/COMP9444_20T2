# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear1 = nn.Linear(784, 10)

    def forward(self, x):
        #return 0 # CHANGE CODE HERE
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = F.log_softmax(out,dim=1)
        
        return out

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        input_size = 784
        hidden1_size = 500
        output_size = 10
        
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.output = nn.Linear(hidden1_size,output_size)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        out = self.fc1(x)
        out = torch.tanh(out)
        output = self.output(out)
        output = F.log_softmax(output,dim=1)
        
        return output # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
      
        self.cnn1 = nn.Conv2d(1, 12, kernel_size = 5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.cnn2 = nn.Conv2d(12, 24, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(384,150)
        self.fc2 = nn.Linear(150,10)


    def forward(self, x):
        #return 0 # CHANGE CODE HERE
        out = self.cnn1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.cnn2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = out.view(out.shape[0],-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        output = F.log_softmax(out,dim=1)
        
        return output
       
