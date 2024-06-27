import warnings

# Suppress UserWarnings from the 'mrinufft' module
warnings.filterwarnings("ignore", category=UserWarning, module="mrinufft")

from data import get_data
from kspace_sampling import get_samples, initialize_2D_cartesian
from physic import corrupt, corrupt_coils
from model import get_model, FISTA, baseline
from plots import plot_rec, plot_img, plot_images_coils
from mrinufft.trajectories.display import display_2D_trajectory
from physic import Nufft

from data import get_images_coils, to_complex_tensor
from fastmri.data.transforms import to_tensor, complex_center_crop, normalize_instance, normalize
from utils import stand, Clip, match_image_stats

import numpy as np
import matplotlib.pyplot as plt
import torch
import fastmri 
import cv2
from skimage.metrics import structural_similarity as ssim
import pdb
from metrics import compute_ssim
from deepinv.optim.data_fidelity import L2
from functools import partial
import argparse

if __name__ == '__main__':

    # samples_loc = get_sparkling()
    choice = 3
    Nc, Ns = 25,100 
    density = 'pipe'

    parser = argparse.ArgumentParser(description="launch experiments")
    parser.add_argument("--traj", type = int, default=3)
    parser.add_argument("--Nc", type = int, default = 50)
    parser.add_argument("--Ns", type = int, default = 1000)
    parser.add_argument("--density", type = str, default = "pipe")
    parser.add_argument("--low_freq", type = bool, default=True)
    parser.add_argument("--model", type = str)
    args = parser.parse_args()

    traj = ['sparkling', 'cartesian', 'radial', 'spiral', 'cones', 'sinusoide', 'propeller', 'rings', 'rosette', 'polar_lissajous', 'lissajous', 'waves']


    samples_loc = get_samples(traj[args.traj], Nc = args.Nc, Ns = args.Ns)
    physic = Nufft((320,320), samples_loc, density=args.density, real=False, Smaps = None)

    target_torch, images, y, y_hat, Smaps, mask = get_data(idx = 0, Smaps = True, physics = physic, low_freq = args.low_freq)
    mask = mask.squeeze(0)

    physic_mcoil = Nufft((320,320), samples_loc, density=args.density, real=False, Smaps = Smaps.squeeze(0).squeeze(0).numpy())
    x0 = physic_mcoil.A_adjoint(y_hat)


    if args.model == "baseline":
        x_hat, data_fidelity_vals, L_x = baseline(images, physic_mcoil, stepsize=None, max_iter = 30, init_norm = True, kspace = y_hat, norm = True)
    elif args.model == "CS":
        x_hat, data_fidelity_vals, prior_vals, L_x = FISTA(images, physic_mcoil, stepsize=None, max_iter = 30, init_norm = True, kspace = y_hat, norm = True, sigma = 1e-4)
    elif args.model == 'PnP':
        iteration = 'PGD'
        algo = get_model(max_iter = 25, stepsize = 1 / physic_mcoil.nufft.get_lipschitz_cst(max_iter = 20), sigma = 1e-4, norm = True, iteration = iteration)
        with torch.no_grad():
            x_hat = algo(y_hat, physic_mcoil, compute_metrics = False)

    plot_rec(target_torch, x_hat, mask, err_with_mask = True, clip = True)


