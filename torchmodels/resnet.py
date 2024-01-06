import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet block configs
_CONV2_X = {18: 2, 34: 3, 50: 3, 101: 3, 152: 3}
_CONV3_X = {18: 2, 34: 4, 50: 4, 101: 4, 152: 8}
_CONV4_X = {18: 2, 34: 6, 50: 6, 101: 23, 152: 36}
_CONV5_X = {18: 2, 34: 3, 50: 3, 101: 3, 152: 3}


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=1, 
                    stride=stride
                ),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        r += self.shortcut(x)
        r = F.relu(r)
        return r


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels*4, 
                kernel_size=1, 
                stride=1, 
                padding=0
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*4)

        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels*4, 
                    kernel_size=1, 
                    stride=stride
                ),
                nn.BatchNorm2d(num_features=out_channels*4)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = F.relu(self.bn2(self.conv2(r)))
        r = self.bn3(self.conv3(r))
        r += self.shortcut(x)
        r = F.relu(r)
        return r


class ResNet(nn.Module):
    def __init__(self, num_layers, in_channels, num_classes):
        super(ResNet, self).__init__()
        assert num_layers in [18, 34, 50, 101, 152], "ResNet: num_layers must be in [18, 34, 50, 101, 152]"

        self.conv1 = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=64, 
                kernel_size=7, 
                stride=2, 
                padding=3
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if num_layers in [18, 34]:
            self.conv2_x = nn.Sequential(
                *[BasicBlock(in_channels=64, out_channels=64) for _ in range(_CONV2_X[num_layers])]
            )
        else:
            self.conv2_x = nn.Sequential(
                BottleneckBlock(in_channels=64, out_channels=64),
                *[BottleneckBlock(in_channels=256, out_channels=64) for _ in range(1, _CONV2_X[num_layers])]
            )

        if num_layers in [18, 34]:
            self.conv3_x = nn.Sequential(
                BasicBlock(in_channels=64, out_channels=128, stride=2),
                *[BasicBlock(in_channels=128, out_channels=128) for _ in range(1, _CONV3_X[num_layers])]
            )
        else:
            self.conv3_x = nn.Sequential(
                BottleneckBlock(in_channels=256, out_channels=128, stride=2),
                *[BottleneckBlock(in_channels=512, out_channels=128) for _ in range(1, _CONV3_X[num_layers])]
            )

        if num_layers in [18, 34]:
            self.conv4_x = nn.Sequential(
                BasicBlock(in_channels=128, out_channels=256, stride=2),
                *[BasicBlock(in_channels=256, out_channels=256) for _ in range(1, _CONV4_X[num_layers])]
            )
        else:
            self.conv4_x = nn.Sequential(
                BottleneckBlock(in_channels=512, out_channels=256, stride=2),
                *[BottleneckBlock(in_channels=1024, out_channels=256) for _ in range(1, _CONV4_X[num_layers])]
            )

        if num_layers in [18, 34]:
            self.conv5_x = nn.Sequential(
                BasicBlock(in_channels=256, out_channels=512, stride=2),
                *[BasicBlock(in_channels=512, out_channels=512) for _ in range(1, _CONV5_X[num_layers])]
            )
        else:
            self.conv5_x = nn.Sequential(
                BottleneckBlock(in_channels=1024, out_channels=512, stride=2),
                *[BottleneckBlock(in_channels=2048, out_channels=512) for _ in range(1, _CONV5_X[num_layers])]
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
                in_features=512 if num_layers in [18, 34] else 2048, 
                out_features=num_classes
        )

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.maxpool(r)
        r = self.conv2_x(r)
        r = self.conv3_x(r)
        r = self.conv4_x(r)
        r = self.conv5_x(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.fc(r)
        return r


def resnet(num_layers, in_channels, num_classes):
    assert num_layers in [18, 34, 50, 101, 152], "Unsupported ResNet model. Supported models are resnet- 18, 34, 50, 101, 152"
    return ResNet(num_layers, in_channels, num_classes)
