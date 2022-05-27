# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:55:01 2021

@author: abdullatif albaseer
"""
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #======= Change model ============
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)#,bidirectional=True
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #=================================
        # the shape we need to our input  x -> (batch_size, sequence, input_size)
        # adding the classification step 
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # RNN input: x and h0 in the correct shape
        # no activation and no softmax at the end 
        # becuase we apply crossentropy loss function will do the softmax
        h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        #h0 = torch.cat((h0[-2,:,:], h0[-1,:,:]), dim = 1) 
        #c0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        out, _ = self.rnn(x,h0)
        #out, _ = self.rnn(x,(h0,c0))
        # batch size, sequ, hidden size
        # out(N, 28, 128)
        # only need the last time step for calssification 
        out = out[:,-1,:]
        out = self.fc(out)# we do the classification depending on the last output of the hidden layer 

        return out


