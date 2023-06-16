import torch.nn as nn
import numpy as np
import torch


class MLP(nn.Module):

    def __init__(self, input_features=26, output_features=4):
        super(MLP, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layer_dimensions = [input_features] + [128, 32] + [output_features]

        self._build_layers()

    def _build_layers(self):
        self.body = nn.ModuleList()
        for i, d in enumerate(self.layer_dimensions[:-1]):
            self.body.append(nn.Linear(in_features=d, out_features=self.layer_dimensions[i + 1]))

    def forward(self, x):
        for op in self.body:
            x = op(x)

        return x


class CNN(nn.Module):

    def __init__(self, H=21, W=128, pooling=False, activation=None, batchnorm=False, dropout_rate=None):
        super(CNN, self).__init__()
        self.H = H
        self.W = W
        self.pooling = pooling
        self.activation_flag = True if activation else False
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'GeLU':
            self.activation = nn.GELU()

        self._build_convolutional_layers()
        self._build_classifier()

    def _build_convolutional_layers(self):
        self.channels = [1, 16, 32, 64, 128]
        convs = []
        for i, in_channels in enumerate(self.channels[:-1]):
            if self.pooling:
                convs.append(
                    nn.Conv2d(in_channels=in_channels, out_channels=self.channels[i + 1], kernel_size=5, padding=2)
                )
            else:
                convs.append(nn.Conv2d(in_channels=in_channels, out_channels=self.channels[i + 1], kernel_size=5))
                self.H = int(np.floor(self.H - 4))
                self.W = int(np.floor(self.W - 4))
            if self.batchnorm:
                convs.append(nn.BatchNorm2d(num_features=self.channels[i + 1]))
            if self.activation_flag:
                convs.append(self.activation)
            if self.pooling:
                convs.append(nn.MaxPool2d(kernel_size=2, padding=1))
                self.H = int(np.floor(self.H / 2 + 1))
                self.W = int(np.floor(self.W / 2 + 1))
        self.convolutional_layers = nn.Sequential(*convs)

    def _build_classifier(self):
        self.layer_dims = [self.H * self.W * self.channels[-1], 1024, 256, 32, 4]
        linears = []
        for i, dims in enumerate(self.layer_dims[:-1]):
            if self.dropout_rate:
                linears.append(self.dropout)
            linears.append(nn.Linear(in_features=dims, out_features=self.layer_dims[i + 1]))
            if self.activation_flag and (i + 1) < len(self.layer_dims[:-1]):
                linears.append(self.activation)
        self.classifier = nn.Sequential(*linears)

    def forward(self, x):

        x = self.convolutional_layers(x)

        return self.classifier(torch.flatten(x, start_dim=1))