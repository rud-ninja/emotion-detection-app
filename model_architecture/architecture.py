import torch
import torch.nn as nn
from torch.nn import functional as F

dropout = 0.35


class residual_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # res layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # res layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # residual path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        self.activation = nn.ReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.activation(out)

        return out


class non_local(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.theta = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        # Following the gaussian embedding method
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        shape = g.shape
        bat, ch, h, w = shape[0], shape[1], shape[2], shape[3]

        # reshaping N, C, H, W to N, C, H*W for matrix multiplication
        theta = theta.view(bat, ch, -1)
        phi = phi.view(bat, ch, -1)
        g = g.view(bat, ch, -1)

        g = torch.transpose(g, 1, 2)

        out = torch.transpose(theta, 1, 2) @ phi
        out = F.softmax(out, dim=-1)

        out = out @ g

        # reinstating N, C H, W shape
        out = torch.transpose(out, 1, 2).view(bat, ch, h, w)

        out = self.W(out)
        out = self.bn(out)

        out = out + x
        out = self.dropout(out)

        return out


class resnet(nn.Module):

    def __init__(self):
        super().__init__()

        n = 64

        # Conv 1
        self.conv1 = nn.Conv2d(1, n, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.conv2 = nn.ModuleList()
        self.conv3 = nn.ModuleList()
        self.conv4 = nn.ModuleList()
        self.conv5 = nn.ModuleList()

        # conv 2 ~ residual block 1
        for ip in [n]*2:
            self.conv2.append(residual_block(ip, ip))

        self.nloc1 = non_local(n)

        # conv 3 ~ residual block 2
        for ip in [n, 2*n]:
            op = 2*n if ip==n else ip
            self.conv3.append(residual_block(ip, op))

        # conv 4 ~ residual block 3
        for ip in [2*n, 4*n]:
            op = 4*n if ip==2*n else ip
            self.conv4.append(residual_block(ip, op))

        # conv 4 ~ residual block 4
        for ip in [4*n, 8*n]:
            op = 8*n if ip==4*n else ip
            self.conv5.append(residual_block(ip, op))


        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # fully connected dense layer
        self.fc = nn.Linear(8*n, 7)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2[0](x)
        x = self.nloc1(x)
        x = self.conv2[1](x)
        x = self.dropout(x)

        for l in self.conv3:
            x = l(x)
        x = self.dropout(x)

        for l in self.conv4:
            x = l(x)
        x = self.dropout(x)

        for l in self.conv5:
            x = l(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)


        return x
