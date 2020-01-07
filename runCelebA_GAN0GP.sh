#!/usr/bin/env bash
python MainImage.py -dataset celeba -ndf 64 -loss gan -arch dcgan -ngf 64 -gp_weight 10 -gp_center 0 -niter 50001 -log_interval 1000 -lrg 2e-4 -lrd 2e-4 -batch_size 64
