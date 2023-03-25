import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


class FacialFeaturesNet(nn.Module):
    def __init__(self):
        super(FacialFeaturesNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x