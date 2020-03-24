import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, num_features, c_num=2, emb_dim=256):
        super(AutoEncoder, self).__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim

        self.encoder = nn.Linear(num_features, 2)
        self.decoder = nn.Linear(2, emb_dim)

    def forward(self, X):
        compressed_feature = self.encoder(X)
        generated_feature = self.decoder(compressed_feature)
        return generated_feature

filters1d = [
    [16, 1],
    [16, 5],
    [32, 9],
    [64, 15],
    [128, 33],
    [256, 71],
]

class Conv1dMovingAverage(nn.Module):
    def __init__(self, num_features, filters):
        super(Conv1dMovingAverage, self).__init__()

        convolutions = []
        for i, (num_channels, num_kernel) in enumerate(filters, start=1):
            same_padding = num_kernel // 2
            conv = nn.Conv1d(
                in_channels=num_features,
                out_channels=num_channels,
                kernel_size=num_kernel,
                padding=same_padding
            )
            convolutions.append(conv)
        self.convolutions = nn.ModuleList(convolutions)
        self.activation = getattr(torch.nn.functional, 'relu')

    def forward(self, X):
        convs = []
        for i in range(len(self.convolutions)):
            convolved = self.convolutions[i](X)
            convolved = self.activation(convolved)
            convs.append(convolved)
        output = torch.cat(convs, dim=1)
        return output

class ConvolutionFeature(nn.Module):
    def __init__(self, num_features, emb_dim=256):
        super(ConvolutionFeature, self).__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim

        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=emb_dim, kernel_size=3, padding=1)

    def forward(self, X):
        c1 = self.conv1(X)
        c2 = self.conv2(c1)
        return c2
