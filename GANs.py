import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torchvision
from Discriminators import *
from Generators import *
from Datasets import *
from Logger import *
import pickle as pkl


def cal_grad_pen(G: Generator, D: Discriminator, real_batch, fake_batch, args):
    # print(args.gp_inter)
    alpha = args.gp_inter
    device = args.device
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_batch.size(0), 1, device=device)
    alpha = alpha.expand(real_batch.size())

    interpolates = alpha * real_batch + ((1 - alpha) * fake_batch)
    interpolates.requires_grad_(True)
    disc_interpolates = D(interpolates)
    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - args.gp_center) ** 2).mean()
    return gradient_penalty


def GAN(G: Generator, D: Discriminator, args):
    D.to(args.device)
    G.to(args.device)

    ones = torch.ones(args.batch_size, 1, device=args.device)
    zeros = torch.zeros(args.batch_size, 1, device=args.device)

    optim_d = None
    optim_g = None
    if args.optimizer == 'adam':
        optim_g = optim.Adam(lr=args.lrg, params=G.parameters(), betas=(args.beta1, args.beta2))
        optim_d = optim.Adam(lr=args.lrd, params=D.parameters(), betas=(args.beta1, args.beta2))
    elif args.optimizer == 'sgd':
        optim_g = optim.SGD(lr=args.lrg, params=G.parameters(), momentum=args.momentum)
        optim_d = optim.SGD(lr=args.lrd, params=D.parameters(), momentum=args.momentum)
    else:
        raise Exception('Not supported optimizer: ' + args.optimizer)

    noise_data = NoiseDataset(distr=args.noise_dist, dim=args.noise_dim, is_image=args.is_image)
    real_data = load_dataset(args.dataset, args)

    criterion = nn.BCELoss()

    z_start = noise_data.next_batch(batch_size=32, device=args.device)
    z_end = noise_data.next_batch(batch_size=32, device=args.device)

    fixed_real = real_data.next_batch(args.nrow * args.ncol, device=args.device)
    fixed_noise = noise_data.next_batch(args.nrow * args.ncol, device=args.device)
    fixed_fakes = []
    fixed_fakes_scores = []
    gen_time = [5000, 10000, 20000, -1]
    ffs_idx = [[] for _ in range(len(gen_time))]
    gen_idx = 0
    noise_direction = torch.rand_like(fixed_real[0], device=args.device) if args.noise_direct else None
    torchvision.utils.save_image(
        fixed_real.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
        args.prefix + '/real.png', nrow=args.nrow, normalize=True)
    noise_range = torch.arange(-args.noise_range, args.noise_range, step=args.noise_step,
                               device='cpu').view(-1).numpy()
    for i in range(len(noise_range)):
        noise_level = noise_range[i]
        # print(noise_level)
        real_noise = fixed_real + noise_level * noise_direction / noise_direction.norm()
        torchvision.utils.save_image(
            real_noise.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
            args.prefix + '/real_noise_%05d.png' % i, nrow=args.nrow, normalize=True)

    for it in range(args.niters):
        if it % 100 == 0:
            print('Iteration %d' % it)

        if it == gen_time[gen_idx]:
            gen_idx += 1
            fixed_fakes.append(G(fixed_noise).data)
            torchvision.utils.save_image(
                fixed_fakes[-1].view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                args.prefix + '/fixed_fake_%05d.png' % it, nrow=1, normalize=True)
            fixed_fakes_scores.append([])

        if it % args.log_interval == 0:
            print('Iteration %d' % it)

            # calculate score for fixed fake
            for ffi in range(len(fixed_fakes)):
                ffs_idx[ffi].append(it)
                fixed_fakes_scores[ffi].append(D(fixed_fakes[ffi]).data.cpu().numpy())
                print(it, fixed_fakes_scores[ffi][-1][1][0], 'hey')

            if args.show_grad:
                disp_grad(G, D, noise_data, real_data, criterion, it, args)
                disp_map(G, D, noise_data, real_data, criterion, it, args)
                disp_path(G, D, noise_data, real_data, z_start, z_end, criterion, it, args)
            if args.show_maxima:
                _, _, scores = compute_extrema(D, fixed_real, noise_direction, args.noise_range, args.noise_step)
                disp_extrema(scores, args.prefix + '/extrema_%05d.pdf' % it,
                             args.noise_range, args.noise_step, args.nrow, args.ncol)
                with open(args.prefix + '/extrema.txt', 'a') as f:
                    f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')

                with torch.no_grad():
                    fixed_fake = G(fixed_noise)
                    torchvision.utils.save_image(
                        fixed_fake.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                        args.prefix + '/fake_%05d.png' % it, nrow=args.nrow, normalize=True)
                    _, _, scores = compute_extrema(D, fixed_fake, noise_direction, args.noise_range, args.noise_step)
                    disp_extrema(scores, args.prefix + '/extrema_fake_%05d.pdf' % it,
                                 args.noise_range, args.noise_step, args.nrow, args.ncol)
                    with open(args.prefix + '/extrema_fake.txt', 'a') as f:
                        f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')

        if it % args.save_model == args.save_model - 1:
            print('Saving model')
            torch.save(G, args.prefix + '/G_%05d.t7' % it)
            torch.save(D, args.prefix + '/D_%05d.t7' % it)

        # train D for nd iterations
        for i in range(args.nd):
            optim_d.zero_grad()

            # train real
            real_batch = real_data.next_batch(args.batch_size, args.device)
            pred_real = D(real_batch)
            loss_real = criterion(pred_real, ones) * args.real_weight
            # train fake
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch).detach()
            pred_fake = D(fake_batch)
            loss_fake = criterion(pred_fake, zeros) * args.fake_weight
            # grad pen
            grad_pen = 0
            if args.gp_weight > 0:
                grad_pen = args.gp_weight * cal_grad_pen(G, D, real_batch.detach(), fake_batch.detach(), args)
            # compute total loss
            loss = loss_real + loss_fake + grad_pen
            # update params
            loss.backward()
            optim_d.step()

        # train G for ng iterations
        for i in range(args.ng):
            optim_g.zero_grad()
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch)
            pred_fake = D(fake_batch)
            loss_g = criterion(pred_fake, ones)
            loss_g.backward()
            optim_g.step()

    # plot scores of fixed_fakes
    nimg = args.nrow * args.ncol
    pkl.dump((gen_time, fixed_fakes, fixed_fakes_scores, ffs_idx), open(args.prefix + '/fixed_fake.t7', 'wb'))
    print(ffs_idx)
    for idx, (gi, ff, ffs) in enumerate(zip(gen_time, fixed_fakes, fixed_fakes_scores)):
        fffig, ffax = plt.subplots(nimg, 1, figsize=(4, 128))
        ffs = np.array(ffs)
        print(ffs.shape)
        for i in range(nimg):
            ffax[i].plot(ffs_idx[idx], ffs[:, i])
            ffax[i].set_ylim(-0.1, 1.1)
        plt.savefig(args.prefix + '/fixed_fake_scores_%05d.pdf' % gi, bbox_inches='tight')

    return G, D


def WGAN(G: Generator, D: Discriminator, args):
    D.to(args.device)
    G.to(args.device)

    ones = torch.ones(args.batch_size, 1, device=args.device)
    zeros = torch.zeros(args.batch_size, 1, device=args.device)

    optim_d = None
    optim_g = None
    if args.optimizer == 'adam':
        optim_g = optim.Adam(lr=args.lrg, params=G.parameters(), betas=(args.beta1, args.beta2))
        optim_d = optim.Adam(lr=args.lrd, params=D.parameters(), betas=(args.beta1, args.beta2))
    elif args.optimizer == 'sgd':
        optim_g = optim.SGD(lr=args.lrg, params=G.parameters(), momentum=args.momentum)
        optim_d = optim.SGD(lr=args.lrd, params=D.parameters(), momentum=args.momentum)
    else:
        raise Exception('Not supported optimizer: ' + args.optimizer)

    noise_data = NoiseDataset(distr=args.noise_dist, dim=args.noise_dim, is_image=args.is_image)
    real_data = load_dataset(args.dataset, args)

    criterion = nn.BCELoss()

    z_start = noise_data.next_batch(batch_size=32, device=args.device)
    z_end = noise_data.next_batch(batch_size=32, device=args.device)
    fixed_real = real_data.next_batch(args.nrow * args.ncol, device=args.device)
    fixed_noise = noise_data.next_batch(args.nrow * args.ncol, device=args.device)
    noise_direction = torch.rand_like(fixed_real[0], device=args.device) if args.noise_direct else None
    if args.noise_dim > 2:  # image data
        torchvision.utils.save_image(
            fixed_real.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
            args.prefix + '/real.png', nrow=args.nrow, normalize=True)
        noise_range = torch.arange(-args.noise_range, args.noise_range, step=args.noise_step,
                                   device='cpu').view(-1).numpy()
        for i in range(len(noise_range)):
            noise_level = noise_range[i]
            # print(noise_level)
            real_noise = fixed_real + noise_level * noise_direction / noise_direction.norm()
            torchvision.utils.save_image(
                real_noise.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                args.prefix + '/real_noise_%05d.png' % i, nrow=args.nrow, normalize=True)

    for it in range(args.niters):
        if it % 100 == 0:
            print('Iteration %d' % it)
        if it % args.log_interval == 0:
            print('Iteration %d' % it)
            if args.show_grad:
                disp_grad(G, D, noise_data, real_data, criterion, it, args)
                disp_map(G, D, noise_data, real_data, criterion, it, args)
                disp_path(G, D, noise_data, real_data, z_start, z_end, criterion, it, args)
            if args.show_maxima:
                _, _, scores = compute_extrema(D, fixed_real, noise_direction, args.noise_range, args.noise_step)
                disp_extrema(scores, args.prefix + '/extrema_%05d.pdf' % it,
                             args.noise_range, args.noise_step, args.nrow, args.ncol)
                with open(args.prefix + '/extrema.txt', 'a') as f:
                    f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')

                with torch.no_grad():
                    fixed_fake = G(fixed_noise)
                    torchvision.utils.save_image(
                        fixed_fake.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                        args.prefix + '/fake_%05d.png' % it, nrow=args.nrow, normalize=True)
                    _, _, scores = compute_extrema(D, fixed_fake, noise_direction, args.noise_range, args.noise_step)
                    disp_extrema(scores, args.prefix + '/extrema_fake_%05d.pdf' % it,
                                 args.noise_range, args.noise_step, args.nrow, args.ncol)
                    with open(args.prefix + '/extrema_fake.txt', 'a') as f:
                        f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')

        if it % args.save_model == args.save_model - 1:
            print('Saving model')
            torch.save(G, args.prefix + '/G_%05d.t7' % it)
            torch.save(D, args.prefix + '/D_%05d.t7' % it)

        # train D for nd iterations
        for p in D.parameters():
            p.requires_grad_(True)
        for i in range(args.nd):
            optim_d.zero_grad()

            # train real
            real_batch = real_data.next_batch(args.batch_size, args.device)
            score_real = D(real_batch).mean()
            # train fake
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch).detach()
            score_fake = D(fake_batch).mean()
            # grad pen
            grad_pen = 0
            if args.gp_weight > 0:
                grad_pen = args.gp_weight * cal_grad_pen(G, D, real_batch.detach(), fake_batch.detach(), args)
            # compute total loss
            loss_d = score_fake - score_real + grad_pen
            # update params
            loss_d.backward()
            optim_d.step()

        # train G for ng iterations
        for p in D.parameters():
            p.requires_grad_(False)
        for i in range(args.ng):
            optim_g.zero_grad()
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch)
            score_fake = D(fake_batch).mean()
            loss_g = -score_fake
            loss_g.backward()
            optim_g.step()

    return G, D
