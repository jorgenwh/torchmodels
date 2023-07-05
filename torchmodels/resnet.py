import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)

    def forward(self, x):
        residual = x
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r += residual
        r = F.relu(r)
        return r


class ResNet(nn.Module):
    def __init__(self, in_channels, in_height, in_width, num_blocks, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.num_classes = num_classes

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.residual_tower = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=1)
        )
        self.fc = nn.Linear(self.in_height * self.in_width, 256)
        self.fc_out = nn.Linear(256, self.num_classes)

    def forward(self, x):
        assert len(x.shape) == 4, "Input shape must be (N, C, H, W)"
        N, C, H, W = x.shape
        assert C == self.in_channels and H == self.in_height and W == self.in_width, "Input shape mismatch"
        
        r = self.conv_block(x)
        r = self.residual_tower(r)
        r = self.conv_bn(r)
        r = r.view(N, -1)
        r = self.fc(r)
        r = F.relu(r)
        r = self.fc_out(r)
        return r
