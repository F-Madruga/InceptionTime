import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
        super(Inception, self).__init__()
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = lambda x: x
            bottleneck_channels = 1
        self.conv_bootleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0]//2,
            bias=False
        )
        self.conv_bootleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1]//2,
            bias=False
        )
        self.conv_bootleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2]//2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm1d(num_features=4*n_filters)
        self.activation = activation
    
    def forward(self, X):
        print('1 - Inception =', X.shape)
        y_bottleneck = self.bottleneck(X)
        print('2 - Inception =', y_bottleneck.shape)
        y_maxpool = self.max_pool(X)
        print('3 - Inception =', y_maxpool.shape)
        y1 = self.conv_bootleneck_1(y_bottleneck)
        print('4 - Inception =', y1.shape)
        y2 = self.conv_bootleneck_2(y_bottleneck)
        print('5 - Inception =', y2.shape)
        y3 = self.conv_bootleneck_3(y_bottleneck)
        print('6 - Inception =', y3.shape)
        y4 = self.conv_maxpool(y_maxpool)
        print('7 - Inception =', y4.shape)
        y = torch.cat([y1, y2, y3, y4], axis=1)
        print('8 - Inception =', y.shape)
        y = self.bn(y)
        print('9 - Inception =', y.shape)
        y = self.activation(y)
        print('10 - Inception =', y.shape)
        return y