import torch

from fastmri.data import SliceDataset
import fastmri
import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri.data.transforms import to_tensor, complex_center_crop
import fastmri.data.transforms as T
from physic import corrupt_coils 
import cv2
from fastmri.data.subsample import MaskFunc

import numpy as np
# from data2 import ClassicDataTransform
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union


path = '/neurospin/optimed/BenjaminLapostolle/fast-mri_smal/'

def get_data(idx = 0, Smaps = False, physics = None):

    # data_transform = T.UnetDataTransform(which_challenge="multicoil")
    data_transform = ClassicDataTransform(which_challenge="multicoil", Smaps = Smaps, physics = physics)
    dataset = mri_data.SliceDataset(
        root=pathlib.Path(path),
        transform=data_transform,
        challenge='multicoil'
    )

    dataloader = torch.utils.data.DataLoader(dataset)
    i = -1
    for batch in dataloader:
        if idx != i:
            i += 1
            continue    
        target_torch, images, y, y_hat, Smaps, mask = batch
        y_hat = torch.cat(y_hat, dim = 0).unsqueeze(0)

        return target_torch, images, y, y_hat, Smaps, mask

def get_images_coils(kspace, target, c_abs = False):

    kspace_torch = to_tensor(kspace)
    images = fastmri.ifft2c(kspace_torch)

    if target is not None:
        crop_size = (target.shape[-2], target.shape[-1])
    else:
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
    
    # check for FLAIR 203
    if images.shape[-2] < crop_size[1]:
        crop_size = (images.shape[-2], images.shape[-2])
    
    images = complex_center_crop(images, crop_size)
    
    # absolute value
    if c_abs:
        images = fastmri.complex_abs(images)
    return images

def to_complex_tensor(tensor):
    # Check if the tensor has the expected shape
    if tensor.shape[-1] != 2:
        raise ValueError("The last dimension of the input tensor must have size 2 to form complex numbers.")
    
    # Split the tensor into real and imaginary parts
    real_part = tensor[..., 0]
    imaginary_part = tensor[..., 1]
    
    # Combine the real and imaginary parts into a complex tensor
    complex_tensor = torch.complex(real_part, imaginary_part)
    
    return complex_tensor







# def get_Smaps(image, images, physics):

#     images_c = to_complex_tensor(images)
#     X, Y, Back = corrupt_coils(images_c, physics)    
#     images_c_2dim = to_tensor(images_c)
#     Smaps = torch.sqrt(images_c_2dim ** 2 / torch.sum(images_c_2dim ** 2, dim = 1))
#     Smaps_c = to_complex_tensor(Smaps)
#     mask = torch.Tensor(produce_mask(image)).unsqueeze(0).repeat(20,1,1).unsqueeze(0)
#     Smaps_c_final = Smaps_c * mask
#     return Smaps_c_final.squeeze(0).numpy()

# def produce_mask(image):
    
#     ### could use Otsu instead of approxiamte quantile 
#     q = np.quantile(image, 0.6)
#     image_bin = image.copy()
#     image_bin[image > q] = 1
#     image_bin[image <= q] = 0
#     contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     brain_mask = cv2.drawContours(np.zeros_like(image_bin.astype(np.uint8)), contours, -1, (255), thickness=cv2.FILLED)
#     return brain_mask


class ClassicDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        Smaps: bool = False,
        physics = None,
        use_seed: bool = True,
        use_abs: bool = False,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.Smaps = Smaps 
        self.physics = physics 
        self.low_freq = True

        

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        y = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # inverse Fourier transform to get zero filled solution
        images_nostand = fastmri.ifft2c(y)
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if images_nostand.shape[-2] < crop_size[1]:
            crop_size = (images_nostand.shape[-2], images_nostand.shape[-2])

        if target is not None:
            target_torch = to_tensor(target)
            target_torch = T.center_crop(target_torch, crop_size)
        else:
            target_torch = torch.Tensor([0])

        images_nostand = T.complex_center_crop(images_nostand, crop_size)
        ### normalization 

        # clip = Clip()
        # clip.quantile(images_nostand, 0.99)
        # images = stand(images_nostand.clone(), clip)
        images = images_nostand.clone()

        if self.physics is None:
            return target_torch, images, y
        
        y_hat = []
        for n_coil in range(y.shape[0]):
            y_hat.append(self.physics.A(T.tensor_to_complex_np(images[n_coil])).squeeze(0).squeeze(0))

        if not(self.Smaps):
            return target_torch, images, y, y_hat

        mask = self.__compute_mask__(images_nostand)
        Smaps = self.compute_Smaps(images, y, mask, low_freq=self.low_freq)
        return target_torch, images, y, y_hat, Smaps, mask
    
    def compute_Smaps(self, images, y, mask, low_freq):
        if low_freq:
            return self.compute_Smaps_low(images, y, mask)
        images_hat = T.tensor_to_complex_np(images)
        # images_hat = np.zeros_like(images[:, :, :, 0], dtype=np.complex64)

        # for n in range(images.shape[0]):
        #     images_hat[n] = self.physics.A_adjoint(y_hat[n])[0,0]
        SOS = np.sum((np.abs(images_hat)**2), axis = 0)
        Smaps = images_hat / np.sqrt(SOS)
        return Smaps * mask
    
    def compute_Smaps_low(self, images, y, mask):

        filter_size = (50, 50)
        y_low = apply_hamming_filter(y, filter_size)
        crop_size = (320,320)
        images_low = T.complex_center_crop(fastmri.ifft2c(y_low), crop_size)
        images_low = T.tensor_to_complex_np(images_low[0])
        SOS = np.sum((np.abs(images_low)**2), axis = 0)
        Smaps_low = images_low / np.sqrt(SOS)
        Smaps_low *= mask
        return Smaps_low

    
    def __compute_mask__(self, images):

        '''
        Args 
        images is the complex images obtained from the full FFT of the kspace y. We take the complete kspace and not the undersampled kspace_hat 
        because we assume we can could access this data experimentally 
        '''
        image = fastmri.complex_abs(images)
        # apply Root-Sum-of-Squares if multicoil data
        image = fastmri.rss(image).numpy()

        ### could use Otsu instead of approxiamte quantile 
        q = np.quantile(image, 0.6)
        image_bin = image.copy()
        image_bin[image > q] = 1
        image_bin[image <= q] = 0
        contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brain_mask = cv2.drawContours(np.zeros_like(image_bin.astype(np.uint8)), contours, -1, (255), thickness=cv2.FILLED)
        brain_mask[brain_mask == 255] = 1
        return brain_mask

# def apply_low_frequency_filter(kspace, filter_size):
#     """
#     Applies a low-frequency filter to the k-space data.
    
#     Parameters:
#     - kspace: A tensor of shape (1, 20, 640, 320, 2) representing the k-space data.
#     - filter_size: A tuple (height, width) specifying the size of the low-frequency filter.
    
#     Returns:
#     - filtered_kspace: The low-frequency filtered k-space data with the same shape as input.
#     """
#     # Combine real and imaginary parts into a complex tensor
#     kspace_complex = torch.view_as_complex(kspace)
    
#     # Create a low-frequency mask
#     mask = torch.zeros_like(kspace_complex)
#     center_h = kspace_complex.shape[2] // 2
#     center_w = kspace_complex.shape[3] // 2
#     h_half = filter_size[0] // 2
#     w_half = filter_size[1] // 2
#     mask[:, :, center_h - h_half:center_h + h_half, center_w - w_half:center_w + w_half] = 1
    
#     # Apply the mask to the k-space data
#     filtered_kspace_complex = kspace_complex * mask
    
#     # Convert back to separate real and imaginary parts
#     filtered_kspace = torch.view_as_real(filtered_kspace_complex)
    
#     return filtered_kspace


def apply_hamming_filter(kspace, filter_size):
    """
    Applies a Hamming window filter to the k-space data.
    
    Parameters:
    - kspace: A tensor of shape (1, 20, 640, 320, 2) representing the k-space data.
    - filter_size: A tuple (height, width) specifying the size of the low-frequency filter.
    
    Returns:
    - filtered_kspace: The Hamming window filtered k-space data with the same shape as input.
    """
    # Combine real and imaginary parts into a complex tensor
    kspace_complex = torch.view_as_complex(kspace).unsqueeze(0) ### change the unsqueeze if necessary
    
    # Create the Hamming window filter
    hamming_window_1d_h = np.hamming(filter_size[0])
    hamming_window_1d_w = np.hamming(filter_size[1])
    hamming_window_2d = np.outer(hamming_window_1d_h, hamming_window_1d_w)
    
    # Pad the Hamming window to the size of the k-space data
    padded_hamming_window = np.zeros((kspace_complex.shape[2], kspace_complex.shape[3]))
    center_h = kspace_complex.shape[2] // 2
    center_w = kspace_complex.shape[3] // 2
    h_half = filter_size[0] // 2
    w_half = filter_size[1] // 2
    
    # Adjust the indices to correctly place the Hamming window at the center
    start_h = center_h - h_half
    end_h = start_h + filter_size[0]
    start_w = center_w - w_half
    end_w = start_w + filter_size[1]
    
    padded_hamming_window[start_h:end_h, start_w:end_w] = hamming_window_2d
    
    # Convert the Hamming window to a PyTorch tensor
    hamming_filter = torch.tensor(padded_hamming_window, dtype=torch.complex64).to(kspace.device)
    
    # Apply the Hamming filter to the k-space data
    filtered_kspace_complex = kspace_complex * hamming_filter
    
    # Convert back to separate real and imaginary parts
    filtered_kspace = torch.view_as_real(filtered_kspace_complex)
    
    return filtered_kspace