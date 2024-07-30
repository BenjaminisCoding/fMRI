r"""Pytorch Dataset for fastMRI.

Code modified from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/mri_data.py

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import random
from pathlib import Path
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Union,
    Sequence,
)
import os
import h5py
import torch
import numpy as np
from deepinv.physics.mri import ifft2c_new
import torchvision
from torchvision import transforms
import json 
from utils import complex_center_crop
from tqdm import tqdm 
import argparse
from utils import virtual_coil_combination_2D
import xml.etree.ElementTree as etree
from kspace_sampling import get_samples
from physic import Nufft
import cv2

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")
    return str(value.text)

class FastMRIIPR(torch.utils.data.Dataset): #IPR stands for Inverse Problem Resolution
    """Dataset for `fastMRI <https://fastmri.med.nyu.edu/>`_ that provides access to MR image slices.

    | 1) The fastMRI dataset includes two types of MRI scans: knee MRIs and
    | the brain (neuro) MRIs, and containing training, validation, and masked test sets.
    | 2) MRIs are volumes (3D) made of slices (2D).
    | 3) This class in particular considers one data sample as one slice of a MRI scan,
    | thus slices of the same MRI scan are considered independently in the dataset.


    **Raw data file structure:** ::

        self.root --- file1000005.h5
                   |
                   -- xxxxxxxxxxx.h5

    | 0) To download raw data, please go to the bottom of the page `https://fastmri.med.nyu.edu/`
    | 1) Each MRI scan is stored in a HDF5 file and can be read with the h5py package.
    | Each file contains the k-space data, ground truth and some meta data related to the scan.
    | 2) MRI scans can either be single-coil or multi-coil with each coil in
    | a multi-coil MRI scan focusses on a different region of the image.
    | 3) In multi-coil MRIs, k-space data has the following shape:
    | (number of slices, number of coils, height, width)
    | 4) For single-coil MRIs, k-space data has the following shape:
    | (number of slices, height, width)

    :param Union[str, Path] root: Path to the dataset.
    :param bool test: Whether the split is the "test" set.
    :param str challenge: "singlecoil" or "multicoil" depending on the type of mri scan.
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param Union[str, Path] metadata_cache_file: A file used to cache dataset
        information for faster load times.
    :param callable, optional sample_filter: A callable object that takes a
        :meth:`SliceSampleFileIdentifier` as input and returns a boolean indicating
        whether the sample should be included in the dataset.
    :param float, optional sample_rate: A float between 0 and 1. This controls what
        fraction of all slices should be loaded. Defaults to 1.
        When creating a sampled dataset either set sample_rate (sample by slices)
        or volume_sample_rate (sample by volumes) but not both.
    :param float, optional volume_sample_rate: A float between 0 and 1. This controls
        what fraction of the volumes should be loaded. Defaults to 1 if no value is given.
        When creating a sampled dataset either set sample_rate (sample by slices)
        or volume_sample_rate (sample by volumes) but not both.
    :param callable, optional transform_kspace: A function/transform that takes in the
        kspace and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    :param callable, optional transform_target: A function/transform that takes in the
        target and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instanciate dataset without transform ::

            from deepinv.datasets import FastMRISliceDataset
            root = "/path/to/dataset/fastMRI/knee_singlecoil/train"
            dataset = FastMRISliceDataset(root=root, test=False, challenge="singlecoil")
            target, kspace = dataset[0]
            print(target.shape)
            print(kspace.shape)

        Instanciate dataset with transform ::

            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop((256, 256), pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]) # Define the transform pipeline
            root = "/path/to/dataset/fastMRI/knee_singlecoil/train"
            dataset = FastMRISliceDataset(root=root, test=False, challenge="multicoil", transform_kspace=transform, transform_target=transform)
            target, kspace = dataset[0]
            print(target.shape)
            print(kspace.shape)

    """

    class SliceSampleFileIdentifier(NamedTuple):
        """Data structure for identifying specific slices within MRI data files."""

        fname: Path
        slice_ind: int
        meta_data: dict

    def __init__(
        self,
        dataset_path: Union[str, Path],
        test: bool,
        challenge: str,
        acceleration_factor: int,
        density: str,
        trajectory: str,
        combine_method: str,
        var_noise: float,
        filter_size_smaps = list,
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        sample_filter: Callable = lambda raw_sample: True,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> None:
        # check that root is a folder
        if not os.path.isdir(dataset_path):
            raise ValueError(
                f"The `root` folder doesn't exist. Please set `root` properly. Current value `{dataset_path}`."
            )
        # check that root folder contains only hdf5 files
        if not all([file.endswith(".h5") for file in os.listdir(dataset_path)]):
            raise ValueError(
                f"The `root` folder doesn't contain only hdf5 files. Please set `root` properly. Current value `{dataset_path}`."
            )
        # ensure that challenge is either singlecoil or multicoil
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('`challenge` should be either "singlecoil" or "multicoil"')
        # ensure that sample_rate and volume_sample_rate are not used simultaneously
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "Either set `sample_rate` (sample by slices) or `volume_sample_rate` (sample by volumes) but not both"
            )
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.dataset_path = dataset_path
        self.challenge = challenge
        self.test = test
        self.trajectory = trajectory 
        self.density = density
        self.acceleration_factor = acceleration_factor
        self.combine_method = combine_method
        self.var_noise = var_noise 
        self.filter_size_smaps = filter_size_smaps

        ### LOAD DATA SAMPLE IDENTIFIERS -----------------------------------------------

        # should contain all the information to load a slice from the storage
        self.sample_identifiers = []
        if load_metadata_from_cache:  # from a cache file
            metadata_cache_file = Path(metadata_cache_file)
            if not metadata_cache_file.exists():
                raise ValueError(
                    "`metadata_cache_file` doesn't exist. Please either deactivate"
                    + "`load_dataset_from_cache` OR set `metadata_cache_file` properly."
                )
            with open(metadata_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
                if dataset_cache.get(dataset_path) is None:
                    raise ValueError(
                        "`metadata_cache_file` doesn't contain the metadata. Please"
                        + "either deactivate `load_dataset_from_cache` OR set `metadata_cache_file` properly."
                    )
                print(f"Using dataset cache from {metadata_cache_file}.")
                self.sample_identifiers = dataset_cache[dataset_path]
        else:
            files = sorted(list(Path(dataset_path).iterdir()))
            for fname in files:
                with h5py.File(fname, "r") as hf:
                    num_slices = hf["kspace"].shape[0]
                    metadata, _ = self._retrieve_metadata(fname)
                    # add each slice to the dataset after filtering
                    for slice_ind in range(num_slices):
                        slice_id = self.SliceSampleFileIdentifier(fname, slice_ind, metadata)
                        if sample_filter(slice_id):
                            self.sample_identifiers.append(slice_id)

            # save dataset metadata
            if save_metadata_to_cache:
                dataset_cache = {}
                dataset_cache[dataset_path] = self.sample_identifiers
                print(f"Saving dataset cache to {metadata_cache_file}.")
                with open(metadata_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)

        ### RANDOM SUBSAMPLING (1 sample = 1 slice from a MRI scan) --------------------

        # set default sampling mode to get the full dataset
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        if sample_rate < 1.0:  # sample by slice / randomly keep a portion of mri slices
            random.shuffle(self.sample_identifiers)
            num_samples = round(len(self.sample_identifiers) * sample_rate)
            self.sample_identifiers = self.sample_identifiers[:num_samples]
        elif (
            volume_sample_rate < 1.0
        ):  # sample by volume / randomly keep a portion of mri scans
            vol_names = list(set([f[0].stem for f in self.sample_identifiers]))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.sample_identifiers = [
                sample_id
                for sample_id in self.sample_identifiers
                if sample_id[0].stem in sampled_vols
            ]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices
    
    def __len__(self) -> int:
        return len(self.sample_identifiers)

    def __getitem__(self, idx: int, mask: Optional[Callable] = None) -> tuple[Any, Any]:
        r"""Returns the idx-th sample from the dataset, both kspace and target.

        The target data is compatible with the physics MRI operator 
        and is a complex tensor of shape (2, H, W).
        """
        fname, dataslice, metadata = self.sample_identifiers[idx]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            Ns, Nc = kspace.shape[-2], kspace.shape[-1]
            if not self.test:
                target = hf[self.recons_key][dataslice]
            else:
                target = None
            image, image_combined, crop_size = self.preprocess(kspace, metadata, target, method = self.combine_method)
            physic, samples_loc = self.define_physic(Nc, Ns, image_combined.shape)
            y = self.get_measurements(image, physic)#The measurements 
            mask = self.get_mask(image_combined)
            smaps = self.get_smaps(kspace, mask, crop_size)
            physic_multicoil = Nufft(
                image_combined.shape, samples_loc, density=self.density, real=False, Smaps = smaps.squeeze(0).squeeze(0).numpy()
                )
        return kspace, image, target, y, smaps, mask, physic, physic_multicoil
    
    def preprocess(self, kspace, metadata, target, method = 'sum'):
        r"""The image is multi-coil, and we aim at combining those coils"""
        assert method in ['sum', 'VCC'], 'method argument should be in [sum]'
        kspace = torch.from_numpy(kspace)
        kspace = torch.view_as_real(kspace)
        image = self.kspace2im(kspace)
        image, crop_size = self.crop(image, metadata, target)
        if method == 'sum':
            image_combined = torch.view_as_complex(image.sum(dim=0))
        if method == "VCC":
            image_combined = torch.view_as_complex(image)
            image_combined = virtual_coil_combination_2D(image_combined)    
        return image, image_combined, crop_size

    def crop(self, image, metadata, target = None):
        r"""
        Remove most of the background of the brain images in the FastMRI dataset.
        Use the metadata to perform the croping operation.
        """
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (metadata["recon_size"][0], metadata["recon_size"][1])
        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)
        return image, crop_size
    
    def define_physic(self, Nc, Ns, shapes_image):
        r"""
        Nc, Ns are needed to compute the parameters of acquisition
        with the acceleration_factor parameter defined. shapes_image 
        is need to compute the samples location
        """
        Nc = int(Nc / self.acceleration_factor)
        samples_loc = get_samples(self.trajectory, Nc = Nc, Ns = Ns)
        physic = Nufft(shapes_image, samples_loc, density=self.density, real=False, Smaps = None)
        return physic, samples_loc
    
    def get_measurements(self, image, physic):
        y = []
        for n_coil in range(image.shape[0]):
            y_coil = physic.A(torch.view_as_complex(image[n_coil])).squeeze(0).squeeze(0)
            y_coil.real += np.random.randn(y_coil.shape[0]) * np.sqrt(self.var_noise)
            y_coil.imag += np.random.randn(y_coil.shape[0]) * np.sqrt(self.var_noise)
            y.append(y_coil)
        return np.array(y)
    
    def get_mask(self, image_combined):
        # normalized_image = cv2.normalize(image_combined.abs().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # normalized_image = normalized_image.astype(np.uint8)
        # ret, image_bin = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # brain_mask = cv2.drawContours(np.zeros_like(image_bin.astype(np.uint8)), contours, -1, (255), thickness=cv2.FILLED)
        # brain_mask[brain_mask == 255] = 1
        # return brain_mask
        magn = image_combined.abs().numpy()
        q = np.quantile(magn, 0.6)
        image_bin = np.zeros_like(image_combined)
        image_bin[magn > q] = 1
        contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brain_mask = cv2.drawContours(np.zeros_like(image_bin.astype(np.uint8)), contours, -1, (255), thickness=cv2.FILLED)
        brain_mask[brain_mask == 255] = 1
        return brain_mask

    def get_smaps(self, kspace, mask, crop_size):
        y_low = apply_hamming_filter(kspace, self.filter_size_smaps)
        images_low = complex_center_crop(self.kspace2im(y_low), crop_size)
        images_low = torch.view_as_complex(images_low)
        SOS = np.sum((np.abs(images_low.numpy())**2), axis = 0)
        Smaps_low = images_low / np.sqrt(SOS)
        Smaps_low *= mask
        return Smaps_low

    def kspace2im(self, kspace: torch.Tensor) -> torch.Tensor:
        r"""Converts kspace to image domain."""
        # return fastmri.ifft2c(kspace)
        return ifft2c_new(kspace)
    

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
    kspace_complex = torch.tensor(kspace, dtype = torch.complex64)
    
    # Create the Hamming window filter
    hamming_window_1d_h = np.hamming(filter_size[0])
    hamming_window_1d_w = np.hamming(filter_size[1])
    hamming_window_2d = np.outer(hamming_window_1d_h, hamming_window_1d_w)
    
    # Pad the Hamming window to the size of the k-space data
    padded_hamming_window = np.zeros((kspace_complex.shape[1], kspace_complex.shape[2]))
    center_h = kspace_complex.shape[1] // 2
    center_w = kspace_complex.shape[2] // 2
    h_half = filter_size[0] // 2
    w_half = filter_size[1] // 2
    
    # Adjust the indices to correctly place the Hamming window at the center
    start_h = center_h - h_half
    end_h = start_h + filter_size[0]
    start_w = center_w - w_half
    end_w = start_w + filter_size[1]
    
    padded_hamming_window[start_h:end_h, start_w:end_w] = hamming_window_2d
    
    # Convert the Hamming window to a PyTorch tensor
    hamming_filter = torch.tensor(padded_hamming_window, dtype=torch.complex64).to(kspace_complex.device)
    
    # Apply the Hamming filter to the k-space data
    filtered_kspace_complex = kspace_complex * hamming_filter
    
    # Convert back to separate real and imaginary parts
    filtered_kspace = torch.view_as_real(filtered_kspace_complex)
    
    return filtered_kspace
