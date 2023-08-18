import torch
import torch.nn as nn
import torch.nn.functional as F


logging.getLogger().setLevel(logging.INFO)


class CNNModel(nn.Module):
    def __init__(
        self, vocab_size, num_classes, dropout=0.1, num_filters=8, kernel_size=10
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(self.vocab_size[0], self.num_filters, self.kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)  # Pooling layer, adjust as needed
        self.final_output = nn.Linear(self.num_filters, self.num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x_act):
        x_act = x_act.float()
        x_act = x_act.permute(
            0, 2, 1
        )  # Rearrange for 1D convolution: (batch, features, seq_len)
        x_act = F.relu(self.conv1(x_act))
        x_act = self.pool(x_act)
        x_act = self.dropout(x_act)
        output = torch.mean(x_act, dim=2)  # Global average pooling
        output = self.final_output(output)
        output = F.softmax(output, dim=1)
        return output
