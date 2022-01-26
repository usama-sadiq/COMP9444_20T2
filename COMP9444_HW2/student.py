#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import pandas as pd

from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem.porter import PorterStemmer


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

##### GLOBALS NEEDED FOR CLASSIFICATION########
embedding_dim = 200
num_hidden_nodes = 70
num_layers = 1
bidirectional = False
dropout = 0.2
output_dim = 5
#############################

#######################################
# pre_process_text_file = open("pre_process_text_file.txt", "w")
#######################################

#######################################
# PUNCTUATIONS
#######################################

punctuations = string.punctuation

#######################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # global pre_process_text_file
    global punctuations

    for i in range(0,len(sample)):
        word = sample[i]
        new_word = ""
        for j in word:
            if j not in punctuations:
                new_word = new_word + j
        sample[i] = new_word
        # pre_process_text_file.write(new_word + "\n")
    
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    return batch

# stopWords = {}
stopWords = stopwords.words('english')
wordVectors = GloVe(name='6B', dim=embedding_dim)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """

    return datasetLabel.to(dtype=torch.long, device = 'cpu') - 1

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    netOutput = torch.argmax(netOutput, dim=1)
    netOutput = torch.add(netOutput, 1)
    return netOutput

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        # Inputs for LSTM: input, (h_0, c_0)
        ## input: (N * L * H(in)) when batch_first = True
        ### The input can also be a packed variable length sequence.
        ## h_0: tensor of shape (D * num_layers, N, H(out)) contains the intial hidden
        ### state of each element in the batch. Defaults to zeros if not provided
        ## c_0: tensor of shape (D * num_layers, N, H(cell)) containing the intial cell
        ### state for each element in the batch. Defaults to zeros if not provided
        # where 
        ##  N = batch size
        ##  L = sequence length
        ##  D = 2 if bidirectional = True otherwise 1
        ##  H(in) = input_size
        ##  H(cell) = hidden_size
        ##  H(out) = proj_size if proj_size > 0 otherwise hidden_size

        global num_layers

        if num_layers == 1:
            self.lstm = tnn.LSTM(input_size = embedding_dim, hidden_size = num_hidden_nodes, num_layers = num_layers, batch_first = True)
            self.fc = tnn.Linear(num_hidden_nodes, output_dim)
        elif num_layers > 1:
            self.lstm = tnn.LSTM(input_size = embedding_dim, hidden_size = num_hidden_nodes, num_layers = num_layers, bidirectional = bidirectional, dropout=dropout, batch_first = True)
            self.fc = tnn.Linear(num_hidden_nodes * 2, output_dim)

    def forward(self, input, length):

        packed_embedded = tnn.utils.rnn.pack_padded_sequence(input, length.cpu(), batch_first=True)

        lstm_output, (hidden, cell) = self.lstm(packed_embedded)

        global num_layers

        if num_layers == 1:
            output_fc_layer = self.fc(hidden[-1, :, :])
        elif num_layers > 1:
            output_fc_layer = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim= 1))

        output_log_softmax = tnn.functional.log_softmax(output_fc_layer, dim = 1)

        return output_log_softmax

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        pass

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.01)