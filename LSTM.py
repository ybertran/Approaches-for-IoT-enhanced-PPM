# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:59:47 2023

@author: u0138175
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

logging.getLogger().setLevel(logging.INFO)

class Model(nn.Module):
    def __init__(self, vocab_size, num_numerical_features, dropout, lstm_size, num_classes):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.num_numerical_features = num_numerical_features
        self.embed_size = 10
        self.lstm_dropout = dropout
        self.lstm_size = lstm_size
        self.hidden_dim = 4
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embed_act = nn.Embedding(self.vocab_size, self.embed_size, padding_idx = 0)
        self.lstm = nn.LSTM(self.embed_size + self.num_numerical_features, self.hidden_dim, num_layers=self.lstm_size,  bidirectional=True,  batch_first=True)
        #self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim,  bidirectional=False,  batch_first=True)
        #self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        #self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.final_output = nn.Linear(2*self.hidden_dim, self.num_classes)

    def forward(self, x_act, x_num, mode):
        x_act_embed_enc = self.embed_act(x_act).to(self.device)
        x_embed_enc = torch.cat([x_act_embed_enc, x_num], dim=2)
        lstm_out, (ht, ct) = self.lstm(x_embed_enc)
        
        #lstm_out1 = lstm_out1.permute(0, 2, 1)  # to make the lstm_out as [batch_size, sequence_length, features]
        #lstm_out1 = self.bn1(lstm_out1) # Apply batch normalization
        #lstm_out1 = lstm_out1.permute(0, 2, 1)  # Permute the dimensions back to the original shape
       
        #lstm_out2,_ = self.lstm2(lstm_out1)

        #lstm_out2 = lstm_out2.permute(0, 2, 1)  # to make the lstm_out as [batch_size, sequence_length, features]
        #lstm_out2 = self.bn2(lstm_out2) # Apply batch normalization
        #lstm_out2 = lstm_out2.permute(0, 2, 1)  # Permute the dimensions back to the original shape

        final_hidden_state = lstm_out[:,-1,:]
        out = self.final_output(final_hidden_state)
        #output_files = self.softmax(out)
        return out