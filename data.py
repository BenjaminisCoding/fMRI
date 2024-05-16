import torch

from fastmri.data import SliceDataset
import fastmri
import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri.data.transforms import to_tensor, complex_center_crop
import fastmri.data.transforms as T
from physic import corrupt_coils 

path = '/neurospin/optimed/BenjaminLapostolle/fast-mri_smal/'

def get_data(idx = 0, to_numpy = True):

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

def get_Smaps(image, images, samples_loc):

    X, Y, Back = corrupt_coils(images, samples_loc)

    ### could use Otsu instead of approxiamte quantile 
    img_ = torch.Tensor(image)
    q = torch.quantile(torch.Tensor(img_),  0.6)
    image_bin = torch.Tensor(img_).clone()
    image_bin[torch.Tensor(img_) > q] = 1 
    image_bin[torch.Tensor(img_) <= q] = 0 
    mask = image_bin.unsqueeze(0)
    mask = mask.repeat(20,1,1).unsqueeze(0)

    Smaps = torch.sqrt(Back ** 2 / torch.sum(Back ** 2, dim = 1)) * mask
    return Smaps

    