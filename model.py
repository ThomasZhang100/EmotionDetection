import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.Conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.Dense1 = nn.Linear(in_features=12*12*64, out_features=128)
        self.Dense2 = nn.Linear(in_features=128, out_features=7)
        self.flatten = nn.Flatten()

    
    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.max_pool2d(x,kernel_size=2)
        x = F.dropout(x, p=0.25)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = F.max_pool2d(x,kernel_size=2)
        x = F.dropout(x, p=0.25)
        x = self.flatten(x)
        x = F.relu(self.Dense1(x))
        x = F.dropout(x, p=0.25)
        x = self.Dense2(x)
        return x 






