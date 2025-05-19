import os
import torch
import torch.nn as nn

# Define the CNN, LSTM, and dense layers
class CNNBlock(nn.Module):
    def __init__(self, input_dim, input_length):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1, stride=1) \
                          if input_dim != 32 else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.bn3(out)
        out = self.relu2(out)
        return out

class CNNDropoutLSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNNDropoutLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=True)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out

class FinalDense(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(FinalDense, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

class RBPSignal(nn.Module):
    def __init__(self, input_dim):
        super(RBPSignal, self).__init__()
        self.seq_cnn = CNNBlock(input_dim=input_dim, input_length=101)
        self.lstm = CNNDropoutLSTM(input_dim=32)
        self.dense = FinalDense(input_dim=64, hidden_dim1=128, hidden_dim2=64, output_dim=1)

    def forward(self, seq_input):
        seq_output = self.seq_cnn(seq_input.permute(0, 2, 1))
        seq_output = seq_output.permute(0, 2, 1)
        lstm_output = self.lstm(seq_output)
        lstm_output = lstm_output.mean(dim=1)  # Take the mean across the sequence dimension
        dense_output = self.dense(lstm_output)
        dense_output = torch.squeeze(dense_output, 1)
        return dense_output
