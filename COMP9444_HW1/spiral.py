# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE

        #Fully connected layer with 2 inputs and num_hid as number of hidden units.
        self.fc1 = nn.Linear(2, num_hid)

        #Output layer with num_hid as the input and 1 output.
        self.output = nn.Linear(num_hid,1)

    def forward(self, input):
        #Input (x,y) is converted polar co-ordinates (r,a) with r = sqrt(x * x + y * y), a = atan2(y,x)
        output_r = torch.sqrt((input[:,0]*input[:,0]) + (input[:,1] * input[:,1]))  # CHANGE CODE HERE
        output_a = torch.atan2(input[:,1],input[:,0])
        output = torch.stack((output_r,output_a),dim=1)
        
        #fed into a fully connected layer with 1 hidden layer using tanhh activation
        hd1_sum = self.fc1(output)
        self.hd1_sum_a = torch.tanh(hd1_sum)

        #Single output using sigmoid activation.
        output_sum = self.output(self.hd1_sum_a)
        self.output_sum_a = torch.sigmoid(output_sum)
        
        return self.output_sum_a

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE

        #Fully connected layer with 2 raw coordinates as input and num_hid as the number of hidden units.
        self.fc1 = nn.Linear(2, num_hid)

        #Fully connected layer with num_hid as the input and output.
        self.fc2 = nn.Linear(num_hid,num_hid)

        #Output layer with num_hid as input and 1 output.
        self.output = nn.Linear(num_hid,1)

    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        #print(input.shape)

        #Input passed to Fully Connected layer.
        hd1_sum = self.fc1(input)

        #Hidden layer 1 sum passed to tanh activation function for layer 1.
        self.hd1_sum_a = torch.tanh(hd1_sum)
        #print(self.hd1_sum_a.shape)

        #Hidden layer 1 activation passed to fully connected layer 2. 
        hd2_sum = self.fc2(self.hd1_sum_a)

        #Hidden layer 2 sum passed to tanh activation function.
        self.hd2_sum_a = torch.tanh(hd2_sum)
        #print(self.hd2_sum_a.shape)

        #Hidden layer 2 activation passed to output layer.
        output_sum = self.output(self.hd2_sum_a)

        #Output layer sum passed to sigmoid activation.
        self.output_sum_a = torch.sigmoid(output_sum)
        #print(self.output_sum_a.shape)
        
        #return 0
        return self.output_sum_a

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE
        Dim_2_of_input = 2

        #Fully connected linear layer with 2 raw coordinates as input and num_hid as number of hidden units.
        self.input = nn.Linear(2,num_hid)

        #Fully connected linear layer with shortcut connections and num_id as number of hidden units.
        self.hd1 = nn.Linear(num_hid + Dim_2_of_input, num_hid)

        #Fully connected linear layer with shortcut connections and num_id as number of hidden units.
        self.output = nn.Linear(num_hid * 2 + Dim_2_of_input,1)

    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        #print(input.shape)

        #Input passed to the input Fully connected linear layer with 2 inputs. Inputs: 2 raw coordinates.
        input_sum = self.input(input)

        #Input layer sum passed to tanh activation.
        self.input_sum_a = torch.tanh(input_sum)

        #ShortCut Implementation between input layer and hidden layer 1.
        input_for_hd1 = torch.cat((self.input_sum_a,input), dim = 1) 
        #print(input_for_hd1.shape)

        #Output from hidden layer 1 is input for hidden layer 2.
        hd1_sum = self.hd1(input_for_hd1)

        #Hidden layer 1 sum passed to tanh activation.
        self.hd1_sum_a = torch.tanh(hd1_sum)

        #Shortcut implementation for connections between input, hidden layer 1 and output layer.
        input_for_output = torch.cat((self.hd1_sum_a,self.input_sum_a,input),dim = 1)

        #Output from hidden layer 1 passed to the output layer.
        output_sum = self.output(input_for_output)

        #Output hidden layer sum passed to sigmoid activation.
        self.output_sum_a = torch.sigmoid(output_sum)

        #save the output and return.
        output = self.output_sum_a
        
        #hd2_sum = self.hd2(input_for_hd2)
        #self.hd2_sum_a = torch.tanh(hd2_sum)
        #
        #
        #
        #
        
        #return 0
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
    #print("Hello There")
    
    #
    #elif type(net).__name__ == 'RawNet':
    #else:

    #Using graph output as a guide to write a method called graph hidden(net, layer, node) 
    #which plots the activavtion (after applying the tanh function) of the hidden node with 
    #the specified number (node) in the specified layer (1 or 2).
    #Note: if net is of type PolarNet, graph_output() only needs to behave correctly when layer is 1.

    #Use this code to generate plots of all the hidden nodes in PolarNet, and all the hidden 
    #nodes in both layers of RawNet and ShortNet and include them in your report.

    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again

        
        #if console --net argument value is PolarNet then only plot the activavtion (after applying the tanh function) 
        #of the hidden node with the specified number (node) in layer 1.
        if type(net).__name__ == 'PolarNet':
            if layer == 1:
                hid = net.hd1_sum_a[:,node]

        #if console --net argument value is RawNet then plot the activavtion (after applying the tanh function) 
        #of the hidden node with the specified number (node) in the specified layer (1 or 2).      
        elif type(net).__name__ == 'RawNet':
            if layer == 1:
                hid = net.hd1_sum_a[:,node]
            else:    
                hid = net.hd2_sum_a[:,node]
        
        #if console --net argument value is ShortNet then plot the activavtion (after applying the tanh function) 
        #of the hidden node with the specified number (node) in the specified layer (1 or 2).      
        else:
            if layer == 1:
                hid = net.input_sum_a[:,node]
            else:
                hid = net.hd1_sum_a[:,node]

        #for prediction any hid value greater than or equal to is considered 1 and vice versa.    
        pred = (hid >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
    
    
    