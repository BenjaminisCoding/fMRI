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
import numpy as np

path = '/neurospin/optimed/BenjaminLapostolle/fast-mri_smal/'

def get_data(idx = 0, to_numpy = True, moments = False):

    data_transform = T.UnetDataTransform(which_challenge="multicoil")
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
        image, reconstruction, mean, std, fname, slice_num, _, kspace = batch
        if to_numpy:
            if moments:
                return reconstruction[0].numpy(), kspace.numpy(), mean, std
            else:
                return reconstruction[0].numpy(), kspace.numpy()
        return reconstruction

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

def get_Smaps(image, images, physics):

    images_c = to_complex_tensor(images)
    X, Y, Back = corrupt_coils(images_c, physics)    
    images_c_2dim = to_tensor(images_c)
    Smaps = torch.sqrt(images_c_2dim ** 2 / torch.sum(images_c_2dim ** 2, dim = 1))
    Smaps_c = to_complex_tensor(Smaps)
    mask = torch.Tensor(produce_mask(image)).unsqueeze(0).repeat(20,1,1).unsqueeze(0)
    Smaps_c_final = Smaps_c * mask
    return Smaps_c_final.squeeze(0).numpy()

def produce_mask(image):
    
    ### could use Otsu instead of approxiamte quantile 
    q = np.quantile(image, 0.6)
    image_bin = image.copy()
    image_bin[image > q] = 1
    image_bin[image <= q] = 0
    contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = cv2.drawContours(np.zeros_like(image_bin.astype(np.uint8)), contours, -1, (255), thickness=cv2.FILLED)
    return brain_mask

    

    