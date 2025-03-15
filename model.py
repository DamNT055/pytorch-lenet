import torch
from torch import nn
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # ANCHOR Layer
        self.conv1 = nn.LazyConv2d(6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.LazyConv2d(16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for idx, layer in self.named_children():
            X = layer(X)
            print(f'{idx:<10} {layer.__class__.__name__:<15} output shape:{X.shape}')
