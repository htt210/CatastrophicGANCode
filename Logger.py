import torch
import torch.autograd as ag
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

from Discriminators import *
from Generators import *
from Datasets import *

# variables
grad_fig, grad_ax = plt.subplots(1, 1, figsize=(4, 4))
plt.draw()
map_fig, map_ax = plt.subplots(1, 1, figsize=(4, 4))
plt.draw()
path_fig, path_ax = plt.subplots(1, 1, figsize=(4, 4))
plt.draw()


def disp_grad(G: Generator, D: Discriminator, noise_data: NoiseDataset, real_data: ToyMissingDataset, criterion, it, args):
    plt.figure(grad_fig.number)
    grad_ax.clear()

    drange = real_data.range * real_data.scale + 0.5
    disprange = drange + 0.1
    n_samples = 2048

    noise_batch = noise_data.next_batch(n_samples, device=args.device)
    fake_batch = G(noise_batch)
    fake_batch = fake_batch.data.cpu().numpy()
    real_batch = real_data.next_batch(n_samples, device=args.device).data.cpu().numpy()

    grad_ax.scatter(real_batch[:, 0], real_batch[:, 1], s=2)
    grad_ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
    grad_ax.set_xlim((-disprange, disprange))
    grad_ax.set_ylim((-disprange, disprange))

    nticks = 21
    noise_batch = torch.rand(nticks * nticks, 2, device=args.device)
    ones = torch.ones(nticks * nticks, 1, device=args.device)

    step = 2 * drange / (nticks - 1)
    for i in range(nticks):
        for j in range(nticks):
            noise_batch[i * nticks + j, 0] = -drange + i * step
            noise_batch[i * nticks + j, 1] = -drange + j * step

    noise_batch.requires_grad_()
    with torch.enable_grad():
        out_batch = D(noise_batch)
        if isinstance(D, Discriminator):
            loss = criterion.forward(out_batch, ones)
            loss.backward()
        else:
            loss = -out_batch.mean()
            loss.backward()

    coord = noise_batch.data.cpu().numpy()
    grad = -noise_batch.grad.cpu().numpy()

    grad_ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])

    grad_fig.savefig(args.prefix + '/grad_%05d.pdf' % it, bbox_inches='tight')

    plt.draw()
    plt.pause(0.1)

    return coord, grad


def disp_map(G: Generator, D: Discriminator, noise_data: NoiseDataset, real_data: ToyMissingDataset, criterion, it, args):
    plt.figure(map_fig.number)
    map_ax.clear()

    drange = real_data.range * real_data.scale + 0.5
    disprange = drange + 0.1

    # noise_batch = noise_data.next_batch(512, device=args.device)
    # fake_batch = G(noise_batch)
    # fake_batch = fake_batch.data.cpu().numpy()
    real_batch = real_data.next_batch(512, device=args.device).data.cpu().numpy()

    map_ax.scatter(real_batch[:, 0], real_batch[:, 1], s=2)
    # path_ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
    map_ax.set_xlim((-disprange, disprange))
    map_ax.set_ylim((-disprange, disprange))

    nticks = 5
    noise_batch = torch.rand(nticks * nticks, 2, device=args.device)
    # ones = torch.ones(nticks * nticks, 1, device=args.device)

    step = 2 * drange / (nticks - 1)
    for i in range(nticks):
        for j in range(nticks):
            noise_batch[i * nticks + j, 0] = -drange + i * step
            noise_batch[i * nticks + j, 1] = -drange + j * step

    fake_batch = G(noise_batch)

    start = noise_batch.data.cpu().numpy()
    ends = fake_batch.data.cpu().numpy()

    for i in range(len(start)):
        map_ax.arrow(start[i, 0], start[i, 1], ends[i, 0] - start[i, 0], ends[i, 1] - start[i, 1],
                     length_includes_head=True, head_width=0.1, fill=True)
    map_ax.scatter(start[:, 0], start[:, 1], s=8, c='r', marker='+')

    plt.savefig(args.prefix + '/map_%05d.pdf' % it, bbox_inches='tight')
    plt.draw()
    plt.pause(0.1)


def build_inter(z_start, z_end, method='lerp', step=1e-4):
    '''
    Build latent codes for interpolation.
    :param z_start:
    :param z_end:
    :param method: linear interpolation or spherical interpolation: lerp | slerp
    :param step:
    :return:
    '''
    if method == 'lerp':
        z_list = []
        direction = z_end - z_start
        n_step = int(1 / step)
        direction = direction / direction.norm(2, dim=1, keepdim=True) * step
        for i in range(n_step):
            z_list.append(z_start)
            z_start = z_start + direction

        return z_list
    elif method == 'slerp':
        z_list = []
        angle = torch.acos((z_end * z_start).sum(dim=1, keepdim=True) /
                           z_end.norm(2, dim=1, keepdim=True) / z_start.norm(2, dim=1, keepdim=True))
        n_step = int(1 / step)
        alpha = angle * step
        for i in range(n_step):
            z = (torch.sin(angle - alpha) * z_start + torch.sin(alpha) * z_end) / torch.sin(angle)
            z_list.append(z)

        return z_list
    else:
        raise Exception('Not supported interpolation method')


def distance(x, y):
    return (y - x).norm(2, dim=1, keepdim=True)


def comp_grad(x, D: Discriminator, args):
    gradients = ag.grad(outputs=D(x), inputs=x,
                        grad_outputs=torch.ones(x.size()).to(args.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    return gradients


def comp_path_length(G: Generator, D: Discriminator, z_list, args):
    with torch.no_grad():
        path_length = torch.zeros(z_list[0].size(0), 1)
        path_list = []
        x_prev = G(z_list[0])
        for i in range(1, len(z_list)):
            x = G(z_list[i])
            if args.path_grad:
                comp_grad(x.detach(), D, args)
            dist = distance(x_prev, x)
            path_length += dist
            path_list.append(dist)
            x_prev = x

        return path_length, path_list


def disp_path(G: Generator, D: Discriminator, noise_data: NoiseDataset, real_data: ToyMissingDataset,
              z_start, z_end, criterion, it, args):
    plt.figure(path_fig.number)
    path_ax.clear()

    drange = real_data.range * real_data.scale + 0.5
    disprange = drange + 0.1
    n_samples = 2048

    noise_batch = noise_data.next_batch(n_samples, device=args.device)
    fake_batch = G(noise_batch)
    fake_batch = fake_batch.data.cpu().numpy()
    real_batch = real_data.next_batch(n_samples, device=args.device).data.cpu().numpy()

    path_ax.scatter(real_batch[:, 0], real_batch[:, 1], s=2)
    # path_ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
    path_ax.set_xlim((-disprange, disprange))
    path_ax.set_ylim((-disprange, disprange))

    # build interpolation path
    z_list = build_inter(z_start=z_start, z_end=z_end, method=args.inter_method, step=args.inter_step)
    x_list = []
    # get the datapoints from latent code
    with torch.no_grad():
        for z in z_list:
            x = G(z)
            x_list.append(x)
    # show interpolation path
    paths = torch.cat(x_list, dim=1).unbind(dim=0)
    colors = cm.rainbow(np.linspace(0, 1, len(paths)))
    for i, path in enumerate(paths):
        path = path.reshape(-1, 2).cpu().numpy()
        path_ax.scatter(path[:, 0], path[:, 1], c=colors[i], marker='+', s=4)
        # path_ax.plot(path[:, 0], path[:, 1], c=colors[i])

    plt.savefig(args.prefix + '/path_%05d.pdf' % it, bbox_inches='tight')
    plt.draw()
    plt.pause(0.1)


def compute_extrema(D: Discriminator, xs, noise=None, noise_range=5, noise_step=0.1):
    # print(xs.size())
    # xsscores = D(xs)
    # print(xsscores)
    gradients = []
    grad_norms = []
    scores = []
    noise_range = torch.arange(-noise_range, noise_range, step=noise_step, device=xs.device).view(-1, 1)
    if xs[0].dim() > 2:
        dims = [1] * xs[0].dim()
        noise_range = noise_range.view(-1, *dims)
    # print(grad_range.size())
    ones = torch.ones_like(noise_range)

    for fx in xs.split(1):
        fx.requires_grad_()
        fs = D(fx)
        if noise is None:
            gradient = ag.grad(outputs=fs, inputs=fx,
                               grad_outputs=torch.ones(fs.size(), device=fs.device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            gradient = noise
        normed_grad = gradient / gradient.norm()
        fx_range = noise_range * normed_grad + ones * fx  # n_range x d_x matrix
        score_range = D(fx_range).squeeze()
        scores.append(score_range.data.cpu().numpy())
        grad_norms.append(gradient.norm().item())
        gradients.append(gradient.data / grad_norms[-1])
        # print('score_range', score_range)
    gradients = torch.stack(gradients, dim=0)
    # print(gradients)
    return gradients, grad_norms, scores


def disp_extrema(scores, outfile, noise_range=5, noise_step=0.1, nrow=8, ncol=8):
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(32, 32))
    noise_range = torch.arange(-noise_range, noise_range, step=noise_step).cpu().numpy()
    # print(grad_range.shape)

    for i in range(len(scores)):
        row = i // ncol
        col = i % nrow
        # if scores[i].min() > 0:
        #     axes[row][col].set_ylim(-0.1, max(1.1, scores[i].max()))
        axes[row][col].plot(noise_range, scores[i], 'r-')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
