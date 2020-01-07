#!/usr/bin/env bash
python MainImage.py -dataset cifar10 -ndf 64 -loss gan -arch dcgan -ngf 64 -gp_weight 100 -gp_center 0 -gp_inter 1 -niter 50001 -log_interval 1000 -lrg 2e-4 -lrd 2e-4 -batch_size 64 -real_weight 1 -fake_weight 1
