import optuna
import argparse

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

import fastmri
from deepinv.optim.data_fidelity import L2
from deepinv.models import WaveletDictDenoiser
from tqdm import tqdm
from deepinv.models import DRUNet
from deepinv.optim import optim_builder, PnP
from model import ComplexDenoiser, get_DPIR_params
from metrics import compute_ssim
from deepinv.utils.metric import cal_psnr

import os 


# Define an objective function to be minimized.
def objective(trial, traj, Nc, Ns, name, norm_init = True, norm_prior = True, **kwargs):

    # Suggest values for the parameters.
    iteration = "FISTA"
    # iteration = "HQS"
    max_iter = 40
    # sigma = trial.suggest_loguniform('sigma', 1e-5, 1e-1)
    # s1 = trial.suggest_categorical('s1', [1,10,100,1000])
    # sigma0 = s1 * sigma
    # lamb = trial.suggest_loguniform('lamb', 1e-2, 1e2)
    sigma = 0.0029578393071253723
    s1 = 10
    sigma0 = s1 * sigma
    lamb = 0.6354935709749694

    # samples_loc = get_samples(traj, Nc = Nc, Ns = Ns)
    # physic = Nufft((320,320), samples_loc, density='pipe', real=False, Smaps = None)
    # target, images, y, y_hat, Smaps, mask = get_data(idx = 0, Smaps = True, physics = physic)
    # physic_mcoil = Nufft((320,320), samples_loc, density='pipe', real=False, Smaps = Smaps.squeeze(0).numpy())

    x0 = physic_mcoil.A_adjoint(y_hat)
    if norm_init:
        x0 = match_image_stats(to_complex_tensor(images[0]), x0)
    stepsize = 1 / physic_mcoil.nufft.get_lipschitz_cst(max_iter = 20)

    data_fidelity = L2()

    # Initialize algo variables
    model = DRUNet(in_channels=1, out_channels=1, pretrained='download').to('cuda')
    model.eval()
    model = ComplexDenoiser(model, norm_prior)
    lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma, max_iter=max_iter, stepsize = stepsize, s1 = sigma0, lamb = lamb)
    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
    early_stop = False  # Do not stop algorithm with convergence criteria
    prior = PnP(denoiser=model)
    algo = optim_builder(
    iteration = iteration,
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=True,
    params_algo=params_algo,
    )

    x_cur = (
        algo.fixed_point.init_iterate_fn(y_hat, physic_mcoil, F_fn= algo.fixed_point.iterator.F_fn)
    if  algo.fixed_point.init_iterate_fn
    else None
    )
    x_true = stand(target)[0]
    x_estimate = stand(x_cur['est'][0].abs())[0]
    L_psnr = [cal_psnr(x_true, x_estimate).item()]
    L_ssim = [compute_ssim(x_true*mask, x_estimate*mask)]
    i = 0 
    while i < max_iter:
        x_cur, _, check_iteration = algo.fixed_point.one_iteration(x_cur, i, y_hat, physic_mcoil, compute_metrics = False, x_gt = None)
        if check_iteration:
            i += 1
        x_estimate = stand(x_cur['est'][0].abs())[0]
        L_psnr.append(cal_psnr(x_true, x_estimate).item())
        L_ssim.append(compute_ssim(x_true*mask, x_estimate*mask))

    # Example objective function: a simple quadratic function.
    res = {
        'psnr_max': np.max(L_psnr),
        'psnr_argmax': np.argmax(L_psnr),
        'ssim_max': np.max(L_ssim),
        'ssim_argmax': np.argmax(L_ssim),
    }
    write_results(res, trial.params, name)
    return np.max(L_psnr)

def write_results(dict, dict_params, name):

    if os.path.exists(f"optuna/optuna_results_{name}.txt"):
        with open(f"optuna/optuna_results_{args.name}.txt", "a") as f:
            f.write('Params/ ')
            for key, value in dict_params.items():
                f.write(f"{key}: {value}/ ")
            f.write('\n')
            f.write('Results/ ')
            for key, value in dict.items():
                f.write(f"{key}: {value}/ ")
            f.write("\n")
    else:
        with open(f"optuna/optuna_results_{args.name}.txt", "w") as f:
            f.write('Params/ ')
            for key, value in dict_params.items():
                f.write(f"{key}: {value}/ ")
            f.write('\n')
            f.write('Results/ ')
            for key, value in dict.items():
                f.write(f"{key}: {value}/ ")
            f.write("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optuna optimization script")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials for the optimization")
    parser.add_argument("--traj", type = int, default=3)
    parser.add_argument("--Nc", type = int, default = 50)
    parser.add_argument("--Ns", type = int, default = 1000)
    parser.add_argument("--name", type = str)
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()

    # Create a study object and specify the direction as 'minimize'.
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='maximize', sampler=sampler)
    traj = ['sparkling', 'cartesian', 'radial', 'spiral', 'cones', 'sinusoide', 'propeller', 'rings', 'rosette', 'polar_lissajous', 'lissajous', 'waves']
    samples_loc = get_samples(traj[args.traj], Nc = args.Nc, Ns = args.Ns)
    physic = Nufft((320,320), samples_loc, density='pipe', real=False, Smaps = None)
    target, images, y, y_hat, Smaps, mask = get_data(idx = 0, Smaps = True, physics = physic)
    physic_mcoil = Nufft((320,320), samples_loc, density='pipe', real=False, Smaps = Smaps.squeeze(0).numpy())

    # Start the optimization process. You can specify the number of trials.
    study.optimize(lambda trial: objective(trial, traj[args.traj], Nc=args.Nc, Ns=args.Ns, name=args.name, n_trials=args.n_trials, n_jobs=args.n_jobs, target = target, images = images, y = y, y_hat = y_hat, Smaps = Smaps, mask = mask, physic_mcoil = physic_mcoil))

    # Print the results.
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    # Write the results to a text file
    with open(f"optuna/optuna_results_{args.name}.txt", "a") as f:
        f.write("Number of finished trials: {}\n".format(len(study.trials)))
        f.write("Best trial:\n")
        f.write("  Value: {}\n".format(trial.value))
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
