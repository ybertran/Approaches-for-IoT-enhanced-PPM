import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.getLogger().setLevel(logging.INFO)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, num_numerical_features, dropout, num_filters, kernel_size, num_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_numerical_features = num_numerical_features
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(self.vocab_size[0]+ self.num_numerical_features, self.num_filters, self.kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)  # Pooling layer, adjust as needed
        self.final_output = nn.Linear(self.num_filters, self.num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x_act , x_num):
        x_act = x_act.float()
        combined_input = torch.cat((x_act, x_num), dim=-1)
        combined_input = combined_input.permute(
            0, 2, 1
        )  # Rearrange for 1D convolution: (batch, features, seq_len)
        combined_input = F.relu(self.conv1(combined_input))
        combined_input = self.pool(combined_input)
        combined_input = self.dropout(combined_input)
        output = torch.mean(combined_input, dim=2)  # Global average pooling
        output = self.final_output(output)
        output = F.softmax(output, dim=1)
        return output