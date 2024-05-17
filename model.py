from deepinv.optim.data_fidelity import L2
from deepinv.optim import optim_builder, PnP
from deepinv.models import DRUNet
from physic import Nufft
import torch
import numpy as np
from physic import corrupt
from deepinv.models import WaveletDictDenoiser
from tqdm import tqdm
from utils import stand

class ComplexDenoiser(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma):
        noisy_batch = torch.cat((x.real, x.imag), 0)
        denoised_batch = self.denoiser(noisy_batch, sigma)
        denoised = denoised_batch[0:1, ...]+1j*denoised_batch[1:2, ...]
        return denoised

def get_model(max_iter = 8, sigma = 0.01):
    
    # Load PnP denoiser backbone
    model = DRUNet(in_channels=1, out_channels=1, pretrained='download')
    model.eval()
    model = ComplexDenoiser(model)
    
    
    # Compute the sequence of parameters along iterates
    def get_DPIR_params(noise_level_img, max_iter=8):
        r"""
        Default parameters for the DPIR Plug-and-Play algorithm.
    
        :param float noise_level_img: Noise level of the input image.
        """
        s1 = 49.0 / 255.0
        s2 = noise_level_img
        # sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        #     np.float32
        # )
        xsi = 0.9
        sigma_denoiser = np.array([max(s1 * (xsi**i) , s2) for i in range(max_iter)]).astype(np.float32)
        stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        lamb = 1 / 0.23
        return lamb, list(sigma_denoiser), list(stepsize), max_iter
    
    
    # Set the DPIR algorithm parameters
    # sigma = 0.01  # Noise level in the image domain
    # max_iter = 8  # Max number of iterations
    lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma, max_iter=max_iter)
    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
    early_stop = False  # Do not stop algorithm with convergence criteria
    
    # Select the data fidelity term
    data_fidelity = L2()
    
    # Specify the denoising prior
    prior = PnP(denoiser=model)
    
    # instantiate the algorithm class to solve the IP problem.
    algo = optim_builder(
        iteration="HQS",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=True,
        params_algo=params_algo,
    )
    return algo

def FISTA(image, physics, stepsize = 0.1, max_iter = 100, Smaps = None, to_stand = False):

    # x, _, _ = corrupt(image, samples_loc)
    x = torch.Tensor(image)
    # Generate the physics
    # physics = Nufft(x[0, 0].shape, samples_loc, density=None, real=False, Smaps = Smaps)
    y = physics.A(x)
    back = physics.A_adjoint(y)
    if stand:
        back = stand(back)

    data_fidelity = L2()
    a = 3  
    sigma = 0.01
    # stepsize = 0.1
       
    # Select a prior
    wav = WaveletDictDenoiser(non_linearity="soft", level=5, list_wv=['db4', 'db8'], max_iter=10)
    device = 'cpu'
    denoiser = ComplexDenoiser(wav).to(device)
    
    # max_iter = 100
    
    # Initialize algo variables
    x_cur = back.clone().to(device)
    w = back.clone().to(device)
    u = back.clone().to(device)
    
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
    
            pbar.set_description(f'Iteration {k}, criterion = {crit:.4f}')
            pbar.update(1)
    
    x_hat = x_cur.clone()
    return x_hat
