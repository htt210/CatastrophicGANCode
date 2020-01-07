import torch
import random
import argparse
import os
import datetime

#seed = 37
#torch.manual_seed(seed)
#random.seed(seed)

from Datasets import *
from Discriminators import *
from Generators import *
from GANs import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-loss', type=str, default='gan', help='loss function: gan | wgan')
    parser.add_argument('-arch', type=str, default='mlp', help='architecture: mlp | dcgan')
    parser.add_argument('-dataset', type=str, default='mnist',
                        help='dataset: mnist | celeba | 8Gaussian | 25Gaussian | Swissroll')
    parser.add_argument('-real_weight', type=float, default=1, help='weight of a real example')
    parser.add_argument('-fake_weight', type=float, default=1, help='weight of a fake example')
    parser.add_argument('-nhidden', type=int, default=512, help='number of hidden unit in MLP')
    parser.add_argument('-nlayer', type=int, default=3, help='number of layer in MLP')
    parser.add_argument('-lrd', type=float, default=3e-4, help='learning rate for D')
    parser.add_argument('-lrg', type=float, default=3e-4, help='learning rate for G')
    parser.add_argument('-nd', type=int, default=1, help='number of D iterations per GAN iteration')
    parser.add_argument('-ng', type=int, default=1, help='number of G iterations per GAN iteration')
    parser.add_argument('-gp_weight', type=float, default=100., help='weight of grad pen')
    parser.add_argument('-gp_center', type=float, default=0., help='grad pen center')
    parser.add_argument('-gp_inter', type=float, default=None,
                        help='grad pen interpolation: 0 | 1 | None <=> on fake | on real | random')
    parser.add_argument('-optimizer', type=str, default='adam', help='optimizer: adam | sgd')
    parser.add_argument('-momentum', type=float, default=0.0, help='momentum for sgd')
    parser.add_argument('-beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('-beta2', type=float, default=0.99, help='beta2 for adam')
    parser.add_argument('-drop_centers', type=str, default=None, help='list of centers to drop, e.g. 1,10,3')

    parser.add_argument('-noise_dist', type=str, default='Gaussian', help='noise distribution: Gaussian')
    parser.add_argument('-noise_dim', type=int, default=50, help='dimensionality of noise distribution')

    parser.add_argument('-niters', type=int, default=200001, help='number of training iteration')
    parser.add_argument('-log_interval', type=int, default=1000, help='log interval')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-stddev', type=float, default=0.02, help='stddev of Gaussians in toy datasets')
    parser.add_argument('-scale', type=float, default=1, help='scale of the toy datasets')
    parser.add_argument('-device', type=str, default='cuda:0', help='device: cuda:x | cpu')

    parser.add_argument('-show_grad', type=bool, default=False, help='show gradients for 2D data')
    parser.add_argument('-show_path', type=bool, default=False, help='show interpolation paths')
    parser.add_argument('-save_image', type=bool, default=True, help='save image for image datasets')
    parser.add_argument('-save_model', type=int, default=100000, help='save model every N iteration')
    parser.add_argument('-show_maxima', type=bool, default=True, help='show f(t) for real datapoints')
    parser.add_argument('-inter_step', type=float, default=1e-2,
                        help='step of the interpolation, 1/inter_step is the number of interpolation steps')
    parser.add_argument('-inter_method', type=str, default='slerp', help='interpolation method: lerp | slerp')
    parser.add_argument('-noise_range', type=float, default=100., help='range of t in f(t)')
    parser.add_argument('-noise_step', type=float, default=1., help='step of t in f(t)')
    parser.add_argument('-noise_direct', type=bool, default=True, help='use gradient for direction in f(t)')
    parser.add_argument('-nrow', type=int, default=8, help='nrow in extrema plot')
    parser.add_argument('-ncol', type=int, default=8, help='ncol in extrema plot')
    parser.add_argument('-is_image', action='store_true', help='work with image')

    args = parser.parse_args()
    print(args)
    prefix = os.path.expanduser('~') + '/github/figs/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix = prefix + args.loss + '_' + args.arch + '_' + args.dataset + '_'

    for k, v in args.__dict__.items():
        if k != 'loss' and k != 'arch' and k != 'dataset' and k != 'device' \
                and k != 'log_interval' and 'show' not in str(k) and 'save' not in str(k) \
                and 'nrow' not in k and 'ncol' not in k:
            prefix += k[0] + k[-1] + str(v) + '_'

    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix += '/' + str(datetime.datetime.now())
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    print(prefix, len(prefix))
    args.prefix = prefix

    image_size = 28
    args.nc = 1
    if args.dataset == 'cifar10':
        image_size = 32
        args.nc = 3
    args.nz = args.noise_dim
    args.nx = image_size * image_size
    args.image_size = image_size

    G = MLPGenerator(nz=args.nz, nx=args.nx, nhidden=args.nhidden, nhiddenlayer=args.nlayer)
    D = MLPDiscriminator(nx=args.nx,  nhidden=args.nhidden, nhiddenlayer=args.nlayer, wgan_output=(args.loss == 'wgan'))

    with open(prefix + '/config.txt', 'w') as f:
        f.write(str(args) + '\n')
        f.write(str(G) + '\n')
        f.write(str(D) + '\n')

    print(G)
    print(D)

    if args.loss == 'gan':
        GAN(G, D, args)
    elif args.loss == 'wgan':
        WGAN(G, D, args)
    else:
        raise Exception('Loss function not supported')
    torch.save(G, prefix + '/G.t7')
    torch.save(D, prefix + '/D.t7')
