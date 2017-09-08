import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class NetA(nn.Module):
    def __init__(self, n_class=10):
        super(NetA, self).__init__()
        # input: 1x28x28
        self.feature = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(32, 64, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.classifier = nn.Linear(64*7*7, n_class)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

class NetB(nn.Module):
    def __init__(self, n_class=10):
        super(NetB, self).__init__()
        # input: 1x28x28
        self.feature = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.classifier = nn.Linear(64*7*7, n_class)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

class NetC(nn.Module):
    def __init__(self, n_class=10):
        super(NetC, self).__init__()
        # input: 1x28x28
        self.feature = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            nn.ReLU()
                        )
        self.classifier = nn.Linear(64*7*7, n_class)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

class NetD(nn.Module):
    def __init__(self, n_class=10):
        super(NetD, self).__init__()
        # input: 1x28x28
        self.feature = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            nn.ReLU()
                        )
        self.classifier = nn.Sequential(
                            nn.Linear(64*7*7, 100),
                            nn.Linear(100, n_class)
                        )


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
