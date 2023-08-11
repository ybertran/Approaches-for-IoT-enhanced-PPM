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
    def __init__(self, vocab_size, dropout, lstm_size, num_classes):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.lstm_dropout = dropout
        self.lstm_size = lstm_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(self.vocab_size[0], self.vocab_size[0], dropout=self.lstm_dropout, num_layers=self.lstm_size, batch_first=True)
        self.final_output = nn.Linear(self.vocab_size[0], self.num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.softmax = F.softmax

    def forward(self, x_act):
        x_act = x_act.float()
        l1_out, _ = self.lstm(x_act)
        output = l1_out[:, -1, :]
        output = self.final_output(output)
        output = self.softmax(output, dim=1)  # Apply softmax activation
        return output

