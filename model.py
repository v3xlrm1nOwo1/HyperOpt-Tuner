import torch
import torch.nn as nn



class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = torch.max_pool2d(out, 2)  
        out = torch.relu(self.conv2(out))
        out = torch.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)  # Dynamically flatten based on batch size
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)
