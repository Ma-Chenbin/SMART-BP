import torch.nn as nn
import torch

"""
(batch_size, channels, sequence_length)
"""


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio

        # Squeeze section: compressing the dimensionality of the feature map
        self.squeeze = nn.AdaptiveAvgPool1d(1)

        # Excitation section: generating feature map weights
        self.excitation = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // self.se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // self.se_ratio, self.in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x).permute(0, 2, 1)  # Compression feature map
        out = self.excitation(out).permute(0, 2, 1)  # Generate feature map weights
        return x * out.expand_as(x)  # Applying weights to the feature map


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.se_block = SEBlock(out_channels, se_ratio=16)  # Adding SE Modules

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se_block(out)  # Application SE Module
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet_se(block, layers, num_classes):
    return ResNet(block, layers, num_classes)


def resnet18_se(num_classes):
    return resnet_se(ResidualBlock, [2, 2, 2, 2], num_classes)


def resnet34_se(num_classes):
    return resnet_se(ResidualBlock, [3, 4, 6, 3], num_classes)
