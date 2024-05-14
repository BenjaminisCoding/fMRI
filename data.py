import torch

from fastmri.data import SliceDataset

import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
import fastmri.data.transforms as T

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
        image, reconstruction, mean, std, fname, slice_num, _ = batch
        if to_numpy:
            return reconstruction[0].numpy()
        return reconstruction
        
    