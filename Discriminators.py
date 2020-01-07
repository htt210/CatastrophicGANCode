import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_conv(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def forward(self, *input):
        pass


class MLPDiscriminator(Discriminator):
    def __init__(self, nx, nhidden, nhiddenlayer, wgan_output=False, negative_slope=1e-2):
        super(MLPDiscriminator, self).__init__()
        self.net = nn.Sequential()
        i = 0
        self.net.add_module('linear_%d' % i, nn.Linear(nx, nhidden))
        self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        for i in range(1, nhiddenlayer):
            self.net.add_module('linear_%d' % i, nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        self.net.add_module('linear_%d' % (i + 1), nn.Linear(nhidden, 1))
        if not wgan_output:
            self.net.add_module('act_%d' % (i + 1), nn.Sigmoid())

        # self.net.apply(weights_init)

    def forward(self, inputs):
        return self.net(inputs)


class DCDiscriminator(Discriminator):
    def __init__(self, img_size, nc, ndf):
        super(DCDiscriminator, self).__init__()
        self.img_size = img_size
        self.nc = nc
        self.ndf = ndf
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init_conv)

    def forward(self, inputs):
        return self.net(inputs)

