import os
import numpy as np
import torch
import random
from torchvision import transforms, datasets, utils
import sklearn.datasets


class NoiseDataset:
    def __init__(self, distr='Gaussian', dim=2, is_image=False):
        self.distr = distr
        self.dim = dim
        self.is_image = is_image

    def next_batch(self, batch_size=64, device=None):
        if self.distr == 'Gaussian':
            if self.is_image:
                return torch.randn(batch_size, self.dim, 1, 1, device=device)
            return torch.randn(batch_size, self.dim, device=device)
        else:
            return 'Not supported distribution'


class ToyMissingDataset:
    def __init__(self, dataset='8Gaussian', stddev=0.02, drop_centers=None, scale=1):
        self.centers = []
        self.stddev = stddev
        self.scale = scale
        self.range = 1
        if dataset == '8Gaussian':
            self.centers = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
        elif dataset == '25Gaussian':
            for x in range(-2, 3):
                for y in range(-2, 3):
                    # point[0] += 2 * x
                    # point[1] += 2 * y
                    self.centers.append((2 * x, 2 * y))
            self.range = 2
        elif dataset == 'Swissroll':
            pass
        self.dataset = dataset
        if drop_centers is not None:
            drop_centers = sorted(drop_centers, reverse=True)
            for center in drop_centers:
                del self.centers[center]

    def next_batch(self, batch_size, device):
        if self.dataset == 'Swissroll':
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            return torch.FloatTensor(data).to(device) * self.scale

        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * self.stddev
            center = random.choice(self.centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        return torch.FloatTensor(dataset).to(device) * self.scale


class MNISTDataset:
    def __init__(self, train=True):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('~/github/data/mnist', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1, shuffle=True)

        self.data = []
        for i, (x, y) in enumerate(self.train_loader):
            self.data.append(x.view(1, -1))
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size, device):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc: self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)


class CIFAR10Dataset:
    def __init__(self, train=True, img_size=32):
        dataloader = torch.utils.data.DataLoader(datasets.CIFAR10(root='~/github/data/cifar10/', download=True,
                                                                  transform=transforms.Compose([
                                                                      transforms.Resize(img_size),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                                                           (0.5, 0.5, 0.5)),
                                                                  ]), train=train), shuffle=True, batch_size=1)
        self.data = []

        self.data = []
        for i, (x, y) in enumerate(dataloader):
            # print(type(x), len(x))
            self.data.append(x)
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size, device):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc: self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)


class FashionMNISTDataset:
    def __init__(self, train=True, img_size=32):
        dataloader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='~/github/data/mnist/', download=True,
                                                                       transform=transforms.Compose([
                                                                           transforms.Resize(img_size),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize((0.5,), (0.5,)),
                                                                       ]), train=train), shuffle=True, batch_size=1)
        self.data = []

        self.data = []
        for i, (x, y) in enumerate(dataloader):
            # print(type(x), len(x))
            self.data.append(x)
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size, device):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc: self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)


class CelebADataset:
    def __init__(self, image_size=64, batch_size=64):
        dataset = datasets.ImageFolder(root=os.path.expanduser('~/github/data/celeba/'),
                                       transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.CenterCrop(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=4, drop_last=True)
        self.iter = iter(self.dataloader)

        # self.data = []
        # for i, x in enumerate(dataloader):
        #     if i % 1000 == 0:
        #         print(i, len(x))
        #     self.data.append(x)
        # self.loc = 0
        # self.n_samples = len(self.data)

    def next_batch(self, batch_size, device):
        # if self.loc + batch_size > self.n_samples:
        #     random.shuffle(self.data)
        #     self.loc = 0
        #
        # batch = self.data[self.loc: self.loc + batch_size]
        # self.loc += batch_size
        # batch = torch.cat(batch, 0)
        batch = next(self.iter, None)
        if batch is None:
            self.iter = iter(self.dataloader)
            batch = next(self.iter, None)

        # print(len(batch), batch.size())
        return batch[0].to(device)


def load_dataset(dataset, args):
    if dataset == 'mnist':
        return MNISTDataset(train=True)
    elif dataset == 'fashionMNIST':
        return FashionMNISTDataset(train=True, img_size=args.image_size)
    elif dataset == 'cifar10':
        return CIFAR10Dataset(train=True, img_size=args.image_size)
    elif dataset == 'celeba':
        return CelebADataset(image_size=args.image_size, batch_size=args.batch_size)
    else:
        return ToyMissingDataset(dataset, stddev=args.stddev,
                                 drop_centers=args.drop_centers, scale=args.scale)
