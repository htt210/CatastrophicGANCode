import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import argparse

from Datasets import *

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
    Adapted from sbarratt
    https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def inception_score_model(generator, noise, n_batch=1000, device='cuda', batch_size=32, resize=True, splits=1):
    N = n_batch * batch_size

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    dtype = torch.cuda.FloatTensor
    generator.to('cuda')
    generator.eval()

    # Set up dataloader
    # dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i in range(n_batch):
        # print(i)
        batch = generator(noise.next_batch(batch_size, device=device))
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]
        # print(i, batch.size(), batch.type())

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch.data)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# if __name__ == '__main__':
#     class IgnoreLabelDataset(torch.utils.data.Dataset):
#         def __init__(self, orig):
#             self.orig = orig
#
#         def __getitem__(self, index):
#             return self.orig[index][0]
#
#         def __len__(self):
#             return len(self.orig)
#
#     import torchvision.datasets as dset
#     import torchvision.transforms as transforms
#
#     cifar = dset.CIFAR10(root='~/github/data/cifar10/', download=True,
#                              transform=transforms.Compose([
#                                  transforms.Scale(32),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                              ])
#     )
#
#     IgnoreLabelDataset(cifar)
#
#     print("Calculating Inception Score...")
#     print(inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))


def inceptions(rootfolder, noise, n_batch=1000, device='cuda', batch_size=32, resize=True, splits=1):
    files = os.listdir(rootfolder)
    for file in files:
        filepath = os.path.join(rootfolder, file)
        if os.path.isdir(filepath):
            inceptions(filepath, noise, n_batch=n_batch, device=device, batch_size=batch_size, resize=resize, splits=splits)
        elif file == 'G.t7':
            print(filepath, inception_score_model(torch.load(filepath), noise, n_batch=n_batch,
                                                  device=device, batch_size=batch_size, resize=resize, splits=splits))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='multi', help='single | multi. Compute inception score of '
                                                                 'single model / all models in a folder')
    parser.add_argument('-gpath', type=str, default='', help='path to generator .t7')
    parser.add_argument('-noise_dim', type=int, default=100, help='noise dim')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n_batch', type=int, default=1000, help='number of batch')
    parser.add_argument('-device', type=str, default='cuda', help='cuda device')
    args = parser.parse_args()

    noise = NoiseDataset(dim=args.noise_dim, is_image=True)
    if args.mode == 'single':
        generator = torch.load(args.gpath)
        print(inception_score_model(generator=generator, noise=noise, batch_size=args.batch_size, n_batch=args.n_batch))
    else:
        inceptions(rootfolder=args.gpath, noise=noise, n_batch=args.n_batch, device=args.device, batch_size=args.batch_size, resize=True, splits=1)
