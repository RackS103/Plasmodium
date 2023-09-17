import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Flatten(),
                                    nn.Linear(56*56*16, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2))

    def forward(self, X):
        return self.model(X)