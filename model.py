from deepinv.optim.data_fidelity import L2
from deepinv.optim import optim_builder, PnP
from deepinv.optim.prior import Zero

from deepinv.models import DRUNet
from physic import Nufft
import torch
import numpy as np
from physic import corrupt
from deepinv.models import WaveletDictDenoiser
from tqdm import tqdm
from utils import stand, Clip, match_image_stats
from data import to_complex_tensor
# from torch.optim import Adam


class ComplexDenoiser(torch.nn.Module):
    def __init__(self, denoiser, norm):
        super().__init__()
        self.denoiser = denoiser
        self.norm = norm

    def forward(self, x, sigma):
        if self.norm:
            x_real, a_real, b_real = stand(x.real)
            x_imag, a_imag, b_imag = stand(x.imag)
        else:
            x_real, x_imag = x.real, x.imag
        noisy_batch = torch.cat((x_real, x_imag), 0)
        # noisy_batch, a, b = stand(noisy_batch)
        noisy_batch = noisy_batch.to('cuda')
        denoised_batch = self.denoiser(noisy_batch, sigma)
        # denoised_batch = denoised_batch * (b -a) + a
        if self.norm:
            denoised = (denoised_batch[0:1, ...] * (b_real - a_real) + a_real)+1j*(denoised_batch[1:2, ...] * (b_imag - a_imag) + a_imag)
        else:
            denoised = denoised_batch[0:1, ...]+1j*denoised_batch[1:2, ...] 
        return denoised.to('cpu')

def get_model(max_iter = 8, sigma = 0.01, s1 =10, lamb = 0.1, stepsize = None, norm = True, **kwargs):
    
    # Load PnP denoiser backbone
    model = DRUNet(in_channels=1, out_channels=1, pretrained='download').to('cuda')
    model.eval()
    model = ComplexDenoiser(model, norm)
    
    
    # Set the DPIR algorithm parameters
    lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma, max_iter=max_iter, stepsize = stepsize, s1 = s1 * sigma, lamb = lamb)
    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}

    early_stop = False  # Do not stop algorithm with convergence criteria
    
    # Select the data fidelity term
    data_fidelity = L2() 
    
    # Specify the denoising prior
    prior = PnP(denoiser=model, **kwargs)
    # prior = Zero()
    # prior = None
    
    # instantiate the algorithm class to solve the IP problem.
    algo = optim_builder(
        # iteration="HQS",
        # iteration="PGD",
        iteration = 'FISTA',
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=True,
        params_algo=params_algo,
        **kwargs,
    )
    return algo

# Compute the sequence of parameters along iterates
def get_DPIR_params(noise_level_img, max_iter, stepsize, s1 = 49.0 / 255.0, lamb = 1 / 0.23):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s2 = noise_level_img
    xsi = 0.7
    sigma_denoiser = np.array([max(s1 * (xsi**i) , s2) for i in range(max_iter)]).astype(np.float32)
    stepsize = np.ones_like(sigma_denoiser) * stepsize
    return lamb, list(sigma_denoiser), list(stepsize), max_iter

def FISTA(images, physics, stepsize = None, max_iter = 100, init_norm = False, kspace = None, norm = False, sigma = 1e-4):

    y = kspace
    back = physics.A_adjoint(y)
    if init_norm:
        back = match_image_stats(to_complex_tensor(images[0]), back) 

    if stepsize is None:
        stepsize = 1 / physics.nufft.get_lipschitz_cst(max_iter = 20)

    data_fidelity = L2()
    a = 3  

    # Select a prior
    wav = WaveletDictDenoiser(non_linearity="soft", level=6, list_wv=['db4', 'db8'], max_iter=15)

    device = 'cuda'
    denoiser = ComplexDenoiser(wav, norm).to(device)
        
    # Initialize algo variables
    x_cur = back.clone()
    w = back.clone()
    u = back.clone()
    
    # Lists to store the data fidelity and prior values
    data_fidelity_vals = []
    prior_vals = []
    L_x = [x_cur.clone()]

    # FISTA iteration
    with tqdm(total=max_iter) as pbar:
        for k in range(max_iter):
    
            tk = (k + a - 1) / a
            tk_ = (k + a) / a
    
            x_prev = x_cur.clone()
    
            x_cur = w - stepsize * data_fidelity.grad(w, y, physics)
            x_cur = denoiser(x_cur, sigma * stepsize)
    
            w = (1 - 1 / tk) * x_cur + 1 / tk * u
    
            u = x_prev + tk * (x_cur - x_prev)
    
            crit = torch.linalg.norm(x_cur.flatten() - x_prev.flatten())

            # Compute and store data fidelity
            data_fidelity_val = data_fidelity(w, y, physics)
            data_fidelity_vals.append(data_fidelity_val.item())

            # Compute and store prior value (for the denoiser)
            prior_val = sigma * stepsize * torch.sum(torch.abs(denoiser(x_cur, sigma * stepsize)))
            prior_vals.append(prior_val.item())
    
            pbar.set_description(f'Iteration {k}, criterion = {crit:.4f}')
            pbar.update(1)
            if k >= 0:
                L_x.append(x_cur)
    
    x_hat = x_cur.clone()
    return x_hat, data_fidelity_vals, prior_vals, L_x

def baseline(images, physics, stepsize = None, max_iter = 100, init_norm = False, kspace = None, norm = False):

    y = kspace
    back = physics.A_adjoint(y)
    if init_norm:
        back = match_image_stats(to_complex_tensor(images[0]), back) ### to better initialize the fista algorithm

    if stepsize is None:
        stepsize = 1 / physics.nufft.get_lipschitz_cst(max_iter = 20)

    data_fidelity = L2()
        
    # Initialize algo variables
    x_cur = back.clone()
    
    # Lists to store the data fidelity and prior values
    data_fidelity_vals = []
    L_x = [x_cur.clone()]

    # FISTA iteration
    with tqdm(total=max_iter) as pbar:
        for k in range(max_iter):
    
            x_prev = x_cur.clone()
            x_cur = x_cur - stepsize * data_fidelity.grad(x_cur, y, physics)

            crit = torch.linalg.norm(x_cur.flatten() - x_prev.flatten())
            # Compute and store data fidelity
            data_fidelity_val = data_fidelity(x_cur, y, physics)
            data_fidelity_vals.append(data_fidelity_val.item())

            # Compute and store prior value (for the denoiser)
            pbar.set_description(f'Iteration {k}, criterion = {crit:.4f}')
            pbar.update(1)
            if k >= 0:
                L_x.append(x_cur)
    
    x_hat = x_cur.clone()
    return x_hat, data_fidelity_vals, L_x

    

