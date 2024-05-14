import torch
from deepinv.physics import LinearPhysics
from mrinufft.density.geometry_based import voronoi

import mrinufft

NufftOperator = mrinufft.get_operator("finufft")

class Nufft(LinearPhysics):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(
        self,
        img_size,
        samples_loc,
        density=None,
        real=False,
        **kwargs
    ):
        super(Nufft, self).__init__(**kwargs)
        
        self.real = real  # Whether to project the data on real images
        if density is not None:
            if density == 'voronoi':
                density = voronoi(samples_loc.reshape(-1, 2))
        
        self.nufft = NufftOperator(samples_loc.reshape(-1, 2), shape=img_size, density=density, n_coils=1, squeeze_dims=False)

    def A(self, x):
        return self.nufft.op(x)

    def A_adjoint(self, kspace):
        return self.nufft.adj_op(kspace)


def corrupt(image, samples_loc):

    image_torch = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = image_torch.clone()
    x = x/x.abs().max() #normalize the data, so it is in the range [0,1]
    
    # Generate the physics
    physics = Nufft(image_torch[0, 0].shape, samples_loc, density=None)
    y = physics.A(x)
    back = physics.A_adjoint(y)
    return x, y, back
    
