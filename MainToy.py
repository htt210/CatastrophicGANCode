import torch
import random
import argparse
import os
import datetime

seed = 37
torch.manual_seed(seed)
random.seed(seed)

from Datasets import *
from Discriminators import *
from Generators import *
from GANs import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-loss', type=str, default='gan', help='loss function: gan | wgan')
    parser.add_argument('-arch', type=str, default='mlp', help='architecture: mlp | dcgan')
    parser.add_argument('-dataset', type=str, default='8Gaussian',
                        help='dataset: mnist | celeba | 8Gaussian | 25Gaussian | Swissroll')
    parser.add_argument('-nhidden', type=int, default=128, help='number of hidden unit in MLP')
    parser.add_argument('-nlayer', type=int, default=2, help='number of layer in MLP')
    parser.add_argument('-lrd', type=float, default=3e-2, help='learning rate for D')
    parser.add_argument('-lrg', type=float, default=1e-2, help='learning rate for G')
    parser.add_argument('-nd', type=int, default=1, help='number of D iterations per GAN iteration')
    parser.add_argument('-ng', type=int, default=1, help='number of G iterations per GAN iteration')
    parser.add_argument('-gp_weight', type=float, default=10., help='weight of grad pen')
    parser.add_argument('-gp_center', type=float, default=0., help='grad pen center')
    parser.add_argument('-gp_inter', type=float, default=1.,
                        help='grad pen interpolation: 0 | 1 | None <=> on fake | on real | random')
    parser.add_argument('-optimizer', type=str, default='sgd', help='optimizer: adam | sgd')
    parser.add_argument('-momentum', type=float, default=0.0, help='momentum for sgd')
    parser.add_argument('-beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('-beta2', type=float, default=0.99, help='beta2 for adam')
    parser.add_argument('-drop_centers', type=str, default=None, help='list of centers to drop, e.g. 1,10,3')

    parser.add_argument('-noise_dist', type=str, default='Gaussian', help='noise distribution: gaussian')
    parser.add_argument('-noise_dim', type=int, default=2, help='dimensionality of noise distribution')

    parser.add_argument('-niters', type=int, default=200001, help='number of training iteration')
    parser.add_argument('-log_interval', type=int, default=1000, help='log interval')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-stddev', type=float, default=0.02, help='stddev of Gaussians in toy datasets')
    parser.add_argument('-scale', type=float, default=1, help='scale of the toy datasets')
    parser.add_argument('-device', type=str, default='cuda:0', help='device: cuda:x | cpu')

    parser.add_argument('-show_grad', action='store_true', help='show gradients for 2D data')
    parser.add_argument('-show_path', action='store_true', help='show interpolation paths')
    parser.add_argument('-save_image', action='store_true', help='save image for image datasets')
    parser.add_argument('-save_model', type=int, default=100000, help='save model after iterations')
    parser.add_argument('-show_maxima', action='store_true', help='show f(t) for real datapoints')
    parser.add_argument('-inter_step', type=float, default=1e-2,
                        help='step of the interpolation, 1/inter_step is the number of interpolation steps')
    parser.add_argument('-inter_method', type=str, default='lerp', help='interpolation method: lerp | slerp')

    args = parser.parse_args()
    print(args)
    prefix = os.path.expanduser('~') + '/github/figs/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix = prefix + args.loss + '_' + args.arch + '_' + args.dataset + '_'

    args.is_image = False

    for k, v in args.__dict__.items():
        if k != 'loss' and k != 'arch' and k != 'dataset' and k != 'device' \
                and k != 'log_interval' and 'show' not in str(k) and 'save' not in str(k) \
                and 'inter_' not in k and 'noise' not in k:
            prefix += k[0] + k[-1] + '_' + str(v) + '_'

    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix += '/' + str(datetime.datetime.now())
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    print(prefix, len(prefix))
    args.prefix = prefix

    G = MLPGenerator(2, 2, args.nhidden, args.nlayer)
    D = MLPDiscriminator(2, args.nhidden, args.nlayer, wgan_output=(args.loss == 'wgan'))

    with open(prefix + '/config.txt', 'w') as f:
        f.write(str(args) + '\n')
        f.write(str(G) + '\n')
        f.write(str(D) + '\n')

    print(G)
    print(D)

    GAN(G, D, args)
    torch.save(G, prefix + '/G.t7')
    torch.save(D, prefix + '/D.t7')
