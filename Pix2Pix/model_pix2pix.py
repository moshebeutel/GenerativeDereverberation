import torch
import torch.nn as nn
import torch.nn.functional as F


# Size of feature maps in generator
ngf = 32
# Size of feature maps in discriminator
ndf = 32


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(7,4), strides=(1,2), padding=1, activation=True, batchnorm=True,
                 bias=True, name=''):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.name = name
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding='valid', bias=bias, dtype=torch.float64)
        self.padding = padding
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, dtype=torch.float64)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = F.pad(x, (0, 0, 3, 3), mode='circular')
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant')
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,4), strides=(1,2), padding=1, activation=True, batchnorm=True,
                 dropout=False,name=''):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.name = name

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding=(1,1), dtype=torch.float64)
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, dtype=torch.float64)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()
        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, ngf, kernel=(7,4), strides=(1, 2), padding=1, batchnorm=False, name='encoder1'),  # bs x 64 x 7 x 64
            DownSampleConv(ngf, ngf * 2, kernel=(7,4), strides=(1, 2), padding=1, name='encoder2'),  # bs x 128 x 7 x 32
            DownSampleConv(ngf * 2, ngf * 4, kernel=(7,4), strides=(1, 2), padding=1, name='encoder3'),  # bs x 256 x 7 x 16
            DownSampleConv(ngf * 4, ngf * 8, kernel=(7,4), strides=(1, 2), padding=1, name='encoder4'),  # bs x 512 x 7 x 8
            DownSampleConv(ngf * 8, ngf * 8, kernel=(7,4), strides=(1, 2), padding=1, name='encoder5'),  # bs x 512 x 7 x 4
            DownSampleConv(ngf * 8, ngf * 8, kernel=(7,4), strides=(1, 2), padding=1, name='encoder6'),  # bs x 512 x 7 x 2
            DownSampleConv(ngf * 8, ngf * 8, kernel=(7,4), strides=(1, 2), padding=1, name='encoder7'),  # bs x 512 x 7 x 1
        ]

        self.decoders = [
            UpSampleConv(ngf * 8, ngf*8, strides=(1, 2), padding=1, dropout=True, name='decoder1'), # bs x 512 x 7 x 2
            UpSampleConv(ngf * 16, ngf*8, strides=(1, 2), padding=1, dropout=True, name='decoder2'), # bs x 512 x 7 x 4
            UpSampleConv(ngf * 16, ngf*8, strides=(1, 2), padding=1, dropout=True, name='decoder3'), # bs x 512 x 7 x 8
            UpSampleConv(ngf * 16, ngf*8, strides=(1, 2), padding=1, dropout=False, name='decoder4'), # bs x 512 x 7 x 16
            UpSampleConv(384, ngf*4, strides=(1, 2), padding=1, dropout=False, name='decoder5'), # bs x 512 x 7 x 32
            UpSampleConv(192, ngf*2, strides=(1, 2), padding=1, dropout=False, name='decoder6'), # bs x 512 x 7 x 64
        ]
        self.final_conv = nn.ConvTranspose2d(2*ngf, out_channels, kernel_size=(3,4), stride=(1,2), padding=(1, 1),dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)
            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        x = self.final_conv(x)
        return self.tanh(x)


class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, ndf, kernel=(7,4), strides=(1, 2), padding=1, bias=False, name='patchgan1')  # channels: 1->64
        self.d2 = DownSampleConv(ndf, ndf * 2, kernel=(7,4), strides=(1, 2), padding=1, bias=False, name='patchgan2')  # channels: 64->128
        self.d3 = DownSampleConv(ndf * 2, ndf * 4, kernel=(7,4), strides=(1, 2), padding=1, bias=False, name='patchgan3')  # channels: 128->256
        self.d4 = DownSampleConv(ndf * 4, ndf * 8, kernel=(7,4), strides=(1, 2), padding=1, bias=False, name='patchgan4')  # channels: 256->512
        self.final = nn.Conv2d(ndf * 8, 1, kernel_size=(7, 4), stride=(1, 1), padding=1, bias=False,dtype=torch.float64)  # channels: 512->1

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
