from metrics import compute_ssim, compute_psnr
import numpy as np
from utils import stand
import matplotlib.pyplot as plt
from deepinv.optim.data_fidelity import L2
from deepinv.optim import optim_builder, PnP
import torch 
from DRUNet_custom import DRUNet
from typing import(
    Dict
)
from deepinv.models import WaveletDictDenoiser


### define the supeclass 

class ModelMRI:

    def __init__(self, physic, y, target, mask, SAVE_PATH, num_epochs, **kwargs) -> None:
        r"""
        self.func_vis_output because the iterator method of the model does not always
        return an output in the right format
        """
        self.target = target
        self.physic = physic
        self.y = y 
        self.stepsize = 1 / physic.nufft.get_lipschitz_cst(max_iter = 20)
        self.target = torch.tensor(target)
        self.mask = mask
        self.save_path = SAVE_PATH
        self.num_epochs = num_epochs 
        self.func_vis_output = lambda x: x.squeeze(0).squeeze(0)

    def __get_x0__(self, y, physic):
        print('Parent init called')
        return torch.tensor(physic.A_adjoint(y))

    def __iteration__(self, x):
        pass

    def run(self):
        self.x0 = self.__get_x0__(self.y, self.physic)
        x0_vis = self.func_vis_output(self.x0)
        L_recon = [x0_vis]
        L_PSNR, L_SSIM = [compute_psnr(self.target, x0_vis)], [compute_ssim(self.target, x0_vis)]
        x = self.x0
        for n in range(1, self.num_epochs+1):
            x = self.__iteration__(x)
            x_vis = self.func_vis_output(x)
            PSNR, SSIM = compute_psnr(self.target, x_vis), compute_ssim(self.target * self.mask, x_vis * self.mask)
            L_recon.append(x_vis)
            L_PSNR.append(PSNR)
            L_SSIM.append(SSIM)
        return L_recon, L_PSNR, L_SSIM 
    
    def plot(self, mask):

        L_recon, L_PSNR, L_SSIM = self.run()
        #### make a figure with two curves, the L_PSNR and the L_SSIM. Add titles and make them beautiful 
        # Plot PSNR and SSIM curves
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(L_PSNR, label='PSNR', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.title(F'PSNR over Epochs. Max : {np.max(L_PSNR): .4f} at {np.argmax(L_PSNR)}')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(L_SSIM, label='SSIM', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.title(F'SSIM over Epochs. Max : {np.max(L_SSIM): .4f} at {np.argmax(L_SSIM)}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path + '/metrics.png')

        # Find the best reconstruction based on PSNR
        idx_PSNR = np.argmax(L_PSNR)
        x_PSNR = L_recon[idx_PSNR].abs().squeeze(0).squeeze(0)
        error = np.abs(stand(self.target.abs())[0]- stand(x_PSNR)[0]) * mask

        # Plot the best reconstruction, error, and target
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(x_PSNR, cmap='gray')
        plt.title('Best Reconstruction (PSNR)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        error_img = plt.imshow(error, cmap='hot')
        plt.title('Error Map')
        plt.axis('off')
        plt.colorbar(error_img, ax=plt.gca(), fraction=0.046, pad=0.04)
        
        plt.subplot(1, 3, 3)
        plt.imshow(self.target, cmap='gray')
        plt.title('Target')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_path + '/brains.png')

class GradientDescent(ModelMRI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_fidelity = L2()
        
    def __iteration__(self, x):
        return x - self.stepsize * self.data_fidelity.grad(x, self.y, self.physic).squeeze(0).squeeze(0)
        

class FISTA(ModelMRI):

    def __init__(
            self, 
            a: int,
            sigma: float,
            wavelet_parameters: Dict,
            **kwargs
            ):
        
        super().__init__(**kwargs)
        self.data_fidelity = L2()
        self.z = self.x0.clone()
        self.u = self.x0.clone()
        self.a = a
        self.sigma = sigma
        self.wavelets = WaveletDictDenoiser(**wavelet_parameters)
        self.denoiser = ComplexDenoiser(self.wavelets, True).to('cuda')
        self.iter = 0 # monitor the iteration
        
    def __iteration__(self, x):
        alpha = (self.iter + self.a - 1) / (self.iter + self.a)
        if not True: ###deepinv version
            x_cur = self.z - self.stepsize * self.data_fidelity.grad(self.z, self.y, self.physic)
            x_cur = self.denoiser(x_cur, self.sigma * self.stepsize)
            self.z = x_cur + alpha * (x_cur - x)
        else:
            x_cur = self.z - self.stepsize * self.data_fidelity.grad(self.z, self.y, self.physic)
            x_cur = self.denoiser(x_cur, self.sigma * self.stepsize)
            self.z  = (1 - 1 / alpha) * x_cur + 1 / alpha * self.u
            u = x + alpha * (x_cur - x)
        self.iter += 1
        return x_cur.squeeze(0).squeeze(0)

class PnPMRI(ModelMRI):

    def __init__(
            self,
            path_weights: str,
            iterator_method: str,
            preconditioner: str,
            sigma: float,
            xsi: float,
            lamb: float,
            start_sigma_factor: float, 
            early_stop: bool = False,
            verbose: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.iter = 0 
        self.func_vis_output = lambda x: (x['est'][0]).squeeze(0).squeeze(0)
        data_fidelity = L2()
        model = self.__load_model__(path_weights)
        model = Denoiser(model)
        params_algo = self.__get_DPIR_params__(
            sigma, 
            start_sigma_factor, 
            self.num_epochs, 
            self.stepsize,
            lamb,
            xsi
        )
        prior = PnP(denoiser=model, **kwargs)
        self.algo = optim_builder(
        iteration = iterator_method,
        preconditioner=preconditioner,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=self.num_epochs,
        verbose=verbose,
        params_algo=params_algo,
        )

    def __get_x0__(self, y, physic):
        return self.algo.fixed_point.init_iterate_fn(torch.tensor(y), physic, F_fn= self.algo.fixed_point.iterator.F_fn)
        
    def __iteration__(self, x):
        with torch.no_grad():
            x = self.algo.fixed_point.single_iteration(
                x, 
                self.iter, 
                self.y, 
                self.physic, 
                compute_metrics = False, 
                x_gt = None
                )
            self.iter += 1
            return x

    def __load_model__(self, path_weights):
        model = DRUNet(in_channels=2, out_channels=2, pretrained=None).to('cuda')
        checkpoint = torch.load(path_weights, map_location=lambda storage, loc: storage)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key.replace("backbone_net.", "")
            new_checkpoint[new_key] = value
        model.load_state_dict(new_checkpoint)
        model.eval()
        return model 

    def __get_DPIR_params__(self, 
            sigma,
            start_sigma_factor, 
            max_iter, 
            stepsize, 
            lamb = 1 / 0.23,
            xsi = 0.97
            ):
        r"""
        Default parameters for the DPIR Plug-and-Play algorithm.

        :param float noise_level_img: Noise level of the input image.
        """
        start_sigma = start_sigma_factor * sigma
        sigma_denoiser = np.array([max(start_sigma * (xsi**i) , sigma) for i in range(max_iter)]).astype(np.float32)
        stepsize = np.ones_like(sigma_denoiser) * stepsize
        params_algo = {
            "stepsize": list(stepsize),
            "g_param": list(sigma_denoiser),
            "lambda": lamb
        }
        return params_algo
    
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

class Denoiser(torch.nn.Module):

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma, norm = True):
        x = torch.permute(torch.view_as_real(x.squeeze(0)), (0,3,1,2)).to('cuda')
        if norm:
            x = x * 1e4
        x_ = torch.permute(self.denoiser(x, sigma).to('cpu'), (0,2,3,1))
        if norm:
            x_ = x_ * 1e-4
        return torch.view_as_complex(x_.contiguous()).unsqueeze(0)


