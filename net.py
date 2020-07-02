#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleConv3Net(torch.nn.Module):
    def __init__(self):
        super(SimpleConv3Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 48 * 5 * 5)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


if __name__ == "__main__":
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 3, 48, 48))
    model = SimpleConv3Net()
    y = model(x)
    print(y)
