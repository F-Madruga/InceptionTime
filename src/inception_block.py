import torch
import torch.nn as nn
from inception import Inception


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), use_residual=True):
        super(InceptionBlock, self).__init__()
        self.activation = activation
        self.use_residual = use_residual
        self.inception_1 = Inception(in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_2 = Inception(in_channels=4*n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_3 = Inception(in_channels=4*n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
            )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                    out_channels=4*n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(num_features=4*n_filters)
            )
    
    def forward(self, X):
        print('1 - Inception Block =', X.shape)
        y = self.inception_1(X)
        print('2 - Inception Block =', y.shape)
        y = self.inception_2(y)
        print('3 - Inception Block =', y.shape)
        y = self.inception_3(y)
        print('4 - Inception Block =', y.shape)
        if self.use_residual:
            y = y + self.residual(X)
            print('5 - Inception Block =', y.shape)
            y = self.activation(y)
            print('6 - Inception Block =', y.shape)
        return y