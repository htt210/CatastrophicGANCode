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


class Generator(nn.Module):
    def forward(self, *input):
        pass


class MLPGenerator(Generator):
    def __init__(self, nz, nx, nhidden, nhiddenlayer, negative_slope=0):
        super(MLPGenerator, self).__init__()
        self.net = nn.Sequential()
        i = 0
        self.net.add_module('linear_%d' % i, nn.Linear(nz, nhidden))
        self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        for i in range(1, nhiddenlayer):
            self.net.add_module('linear_%d' % i, nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        self.net.add_module('linear_%d' % (i + 1), nn.Linear(nhidden, nx))
        # self.net.apply(weights_init)

    def forward(self, inputs):
        return self.net(inputs)


class DCGenerator(Generator):
    def __init__(self, nz, ngf, img_size, nc):
        super(DCGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.img_size = img_size
        self.nc = nc

        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init_conv)

    def forward(self, inputs):
        return self.net(inputs)
