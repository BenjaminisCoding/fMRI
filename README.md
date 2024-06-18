# fMRI Reconstruction

This repository provides tools and techniques for performing reconstruction on the fastMRI dataset. To use this repository, you need access to the fastMRI dataset and must specify its path in the `data.py` file.

The easiest way to start experiments is by working on the notebook `example.ipynb`.

## Quick Overview of the Files

* **model.py**
  - Contains the pipelines to load models and build the reconstruction.

* **data.py**
  - Contains the necessary functions to load the data.

* **kspace_sampling.py**
  - Contains functions to obtain the sampling locations in k-space, based on the `mri-Nufft` package.

* **plots.py**
  - Contains functions to plot the results.

* **physic.py**
  - Integrates `deepinv` library and `mri-nufft` tools.

* **optuna_.py**
  - Facilitates fine-tuning of parameters using the Optuna library. Note that this code is not up to date with the latest release of `deepinv`.

* **metrics.py**
  - Computes metrics to evaluate the performance of the reconstruction, mainly PSNR and SSIM.
