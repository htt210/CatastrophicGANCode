import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision

from Datasets import *

import os
import argparse
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self, nz, nx, nhidden, nlayer):
        super(Generator, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_%d' % 0, nn.Linear(nz, nhidden))
        self.net.add_module('act_%d' % 0, nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        for i in range(1, nlayer + 1):
            self.net.add_module('linear_%d' % i, nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.net.add_module('linear_%d' % i, nn.Linear(nhidden, nx))

    def forward(self, inputs):
        return self.net(inputs)


class ContinualDiscriminator(nn.Module):
    def __init__(self, nx, nhidden, nlayer):
        super(ContinualDiscriminator, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_%d' % 0, nn.Linear(nx, nhidden))
        self.net.add_module('act_%d' % 0, nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        for i in range(1, nlayer + 1):
            self.net.add_module('linear_%d' % i, nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.net.add_module('linear_%d' % i, nn.Linear(nhidden, 1))
        self.net.add_module('act_%d' % i, nn.Sigmoid())

    def forward(self, inputs):
        return self.net(inputs)

    def compute_fisher(self, buffer_real, buffer_fake, batch_mode, device='cuda:0'):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        batch_size = buffer_real[0].size(0)
        zeros = torch.zeros(batch_size, device=device)
        ones = torch.ones(batch_size, device=device)
        for x in buffer_real:
            x = x.to(device)
            loglikelihoods.append(
                F.binary_cross_entropy(self(x), ones, reduction='mean' if batch_mode else None).view(-1, 1)
            )
        for x in buffer_fake:
            x = x.to(device)
            loglikelihoods.append(
                F.binary_cross_entropy(self(x), zeros, reduction='mean' if batch_mode else None).view(-1, 1)
            )

        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def update_fisher(self, fisher, discount):
        '''
        This function is written on the spirit of "Generative Adversarial Network
        Training is a Continual Learning Problem". We could try another formulation
        later.
        :param fisher:
        :param discount:
        :return:
        '''
        with torch.no_grad():
            for n, p in self.named_parameters():
                n = n.replace('.', '__')
                param_sum = getattr(self, '{}.param_sum'.format(n), None)
                fisher_sum = getattr(self, '{}._fisher_sum'.format(n), None)

                if param_sum is None:
                    param_sum_current = fisher[n] * p.data
                    self.register_buffer('{}_param_sum'.format(n),  param_sum_current.data.clone())
                    self.register_buffer('{}_fisher_sum'.format(n), fisher[n].data.clone())
                    self.register_buffer('{}_param_opt'.format(n), p.data.clone())
                else:
                    param_sum = param_sum * discount + fisher[n] * p.data.clone()
                    fisher_sum = fisher_sum * discount + fisher[n]

                    self.register_buffer('{}_param_sum'.format(n), param_sum.data.clone())
                    self.register_buffer('{}_fisher_sum'.format(n), fisher_sum.data.clone())
                    self.register_buffer('{}_param_opt'.format(n), (param_sum / fisher_sum).data.clone())

    def ewc_loss(self):
        loss = 0
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            param_opt = getattr(self, '{}_param_opt'.format(n), None)
            fisher_sum = getattr(self, '{}_fisher_sum'.format(n), None)

            if param_opt is not None:
                # param_optim = param_sum / fisher_sum
                loss += (fisher_sum * ((p.data - param_opt) ** 2)).sum()
            else:
                print('Fisher buffer is None')
                return 0

        return loss


def cal_gradpen(netD, real_data, fake_data, center=0, alpha=None, LAMBDA=1, device=None):
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
    return gradient_penalty


def check_extrema(D, fixed_xs, grad_range=5, grad_step=0.1):
    gradients = []
    grad_norms = []
    scores = []
    grad_range = torch.arange(-grad_range, grad_range, step=grad_step, device=fixed_xs.device).view(-1, 1)
    # print(grad_range.size())
    ones = torch.ones_like(grad_range)

    for fx in fixed_xs.unbind():
        fx.requires_grad_()
        fs = D(fx)
        gradient = autograd.grad(outputs=fs, inputs=fx,
                           grad_outputs=torch.ones(fs.size(), device=fs.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
        fx_range = grad_range * gradient + ones * fx  # n_range x d_x matrix
        score_range = D(fx_range)
        scores.append(score_range.data.cpu().numpy())
        grad_norms.append(gradient.norm().item())
        gradients.append(gradient.data / grad_norms[-1])
        # print('score_range', score_range)
    gradients = torch.stack(gradients, dim=0)
    # print(gradients)
    return gradients, grad_norms, scores


def plot_grid(scores, output, grad_range=5, grad_step=0.1):
    fig, axes = plt.subplots(8, 8, figsize=(32, 32))
    grad_range = torch.arange(-grad_range, grad_range, step=grad_step, device='cuda').cpu().numpy()
    # print(grad_range.shape)

    for i in range(len(scores)):
        row = i // 8
        col = i % 8
        axes[row][col].set_ylim(0., 1.)
        axes[row][col].plot(grad_range, scores[i])
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)


def GAN_GP(D, G, data, noise, nc, img_size, niter=10000, d_steps=1, batch_size=32,
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None,
           device='cuda', prefix='figs/', args=None):
    D.to(device)
    G.to(device)
    optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
    optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))

    criterion = nn.BCELoss()

    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fixed_z = noise.next_batch(64, device=device)
    fixed_real = data.next_batch(64, device=device)
    inter_z = torch.tensor(fixed_z.data)

    torchvision.utils.save_image(fixed_real.resize(64, nc, img_size, img_size), prefix + '/%s_real.png' % args.dataset)

    buffer_real = []
    buffer_fake = []

    for i in range(8):
        for j in range(8):
            inter_j = j / 8.0
            inter_z[i * 8 + j] = (1 - inter_j) * fixed_z[i * 2] + inter_j * fixed_z[i * 2 + 1]

    start = datetime.datetime.now()
    for iter in range(niter):
        if iter % 100 == 0:
            print(datetime.datetime.now() - start, iter)

        if iter % 1000 == 0:
            print(datetime.datetime.now() - start, iter, 'logging')
            with torch.no_grad():
                fake_batch = G(fixed_z)
                imgs = fake_batch.data.resize(64, nc, img_size, img_size)
                torchvision.utils.save_image(imgs, prefix + '/%s_iter_%06d.png' % (args.dataset, iter))

                inter_batch = G(inter_z)
                inter_imgs = inter_batch.data.resize(64, nc, img_size, img_size)
                torchvision.utils.save_image(inter_imgs,
                                             prefix + '/%s_interpolation_iter_%06d.png' % (args.dataset, iter), nrow=8)

            grad_range = 100.
            grad_step = 1.
            grads, grad_norms, scores = check_extrema(D, G(fixed_z), grad_range=grad_range, grad_step=grad_step)
            plot_grid(scores, prefix + 'scores_fake_%06d.pdf' % iter, grad_range=grad_range, grad_step=grad_step)

            grads, grad_norms, scores = check_extrema(D, fixed_real, grad_range=grad_range, grad_step=grad_step)
            plot_grid(scores, prefix + 'scores_real_%06d.pdf' % iter, grad_range=grad_range, grad_step=grad_step)
            # grads = grads.resize(64, nc, img_size, img_size)
            # torchvision.utils.save_image(grads, prefix + '/%s_iter_%06d_grads.png' % (args.dataset, iter), nrow=8)

        # train D
        for i in range(d_steps):
            optim_d.zero_grad()
            real_batch = data.next_batch(batch_size, device=device)
            # real_batch = real_batch.to(device)
            predict_real = D(real_batch)
            loss_real = criterion.forward(predict_real, ones)

            noise_batch = noise.next_batch(batch_size, device=device)
            # noise_batch = noise_batch.to(device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.detach()
            predict_fake = D(fake_batch)
            loss_fake = criterion.forward(predict_fake, zeros)

            gradpen = 0
            if LAMBDA > 0:
                gradpen = cal_gradpen(D, real_batch.detach(), fake_batch.detach(),
                                      center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)

            loss_ewc = 0
            if args.ewcweight > 0:
                loss_ewc = args.ewcweight * D.ewc_loss()

            loss_d = loss_real + loss_fake + gradpen + loss_ewc
            # loss_d = loss_real + loss_fake + gradpen
            loss_d.backward()
            optim_d.step()

            buffer_real.append(real_batch.detach())
            buffer_fake.append(fake_batch.detach())

        if iter % args.interval == args.interval - 1:
            # print('compute %d' % iter, datetime.datetime.now() - start)
            fisher = D.compute_fisher(buffer_real, buffer_fake, args.batch_mode, device=device)
            D.update_fisher(fisher, discount=args.discount)
            buffer_fake = []
            buffer_real = []

        # train G
        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        predict_fake = D(fake_batch)
        loss_g = criterion.forward(predict_fake, ones)
        loss_g.backward()
        optim_g.step()

    return D, G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhidden', type=int, default=512, help='number of hidden neurons')
    parser.add_argument('--nlayers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--niters', type=int, default=20000, help='number of iterations')
    parser.add_argument('--device', type=str, default='cuda', help='id of the gpu. -1 for cpu')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--center', type=float, default=0., help='gradpen center')
    parser.add_argument('--LAMBDA', type=float, default=0., help='gradpen weight')
    parser.add_argument('--alpha', type=float, default=None,
                        help='interpolation weight between reals and fakes. '
                             '1 for real only, 0 for fake only, None for random interpolation')
    parser.add_argument('--lrg', type=float, default=3e-4, help='lr for G')
    parser.add_argument('--lrd', type=float, default=3e-4, help='lr for D')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to use: mnist | stacked')
    parser.add_argument('--loss', type=str, default='gan', help='gan | wgan')
    parser.add_argument('--nz', type=int, default=50, help='dimensionality of noise')
    parser.add_argument('--ncritic', type=int, default=1,
                        help='number of critic/discriminator iterations per generator iteration')
    parser.add_argument('--batch_mode', type=bool, default=True,
                        help='compute Fisher inforation in batch or for each individual sample')
    parser.add_argument('--discount', type=float, default=0.9, help='discount factor for old Fisher info')
    parser.add_argument('--ewcweight', type=float, default=10., help='weight of the ewc loss term')
    parser.add_argument('--interval', type=int, default=100, help='interval for ewc')

    args = parser.parse_args()
    nz = args.nz
    nc = 1
    img_size = 28
    if args.dataset == 'mnist':
        data = MNISTDataset()
    elif args.dataset == 'stacked':
        nc = 3
        data = StackedMNISTDataset()
    nx = img_size * img_size * nc

    prefix = 'figs/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    prefix = 'figs/%s_%s_center_%.2f_alpha_%s_lambda_%.2f_lrg_%.5f_lrd_%.5f_nhidden_%d_nz_%d_discount' \
             '_%.4f_batch_%d_ewcweight_%.2f_interval_%d_ncritic_%d/' % \
             (args.loss, args.dataset, args.center, str(args.alpha), args.LAMBDA, args.lrg, args.lrd, args.nhidden,
              args.nz, args.discount, args.batch_mode, args.ewcweight, args.interval, args.ncritic)
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    prefix = prefix + str(datetime.datetime.now()).replace(' ', '_') + '/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    print(prefix)

    G = Generator(nz, nx, args.nhidden, args.nlayers)
    if args.loss == 'gan':
        D = ContinualDiscriminator(nx, args.nhidden, args.nlayers)
    else:
        raise Exception('Only GAN loss is supported')
    print('D')
    print(D)
    noise = NoiseDataset(dim=nz)

    config = str(args) + '\n' + str(G) + '\n' + str(D) + '\n'
    with open(prefix + 'config.txt', 'w') as f:
        f.write(config)

    if args.loss == 'gan':
        print(args.loss)
        D, G = GAN_GP(D, G, data, noise, nc=nc, img_size=img_size, niter=args.niters + 1, d_steps=args.ncritic,
                      batch_size=args.batch_size, lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA,
                      alpha=args.alpha, device=args.device, prefix=prefix, args=args)

    # torch.save(D, prefix + 'D.t7')
    # torch.save(G, prefix + 'G.t7')
