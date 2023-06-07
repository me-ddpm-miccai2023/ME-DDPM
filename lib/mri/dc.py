import numpy as np
import torch
import torch.nn as nn
from lib.mri.mri_math import fft2c as fft, ifft2c as ifft

"""
Taken from Jo Schlemper, https://github.com/js3611/Deep-MRI-Reconstruction/
Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D. A Deep
Cascade of Convolutional Neural Networks for MR Image Reconstruction.
Information Processing in Medical Imaging (IPMI), 2017
"""

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = 'ortho'
        self.noise_lvl = noise_lvl

    # def forward(self, *input, **kwargs):
    #     return self.perform(*input)

    def forward(self, x, k0, mask):
        """Performs data consistency

        Args:
            x ([float32]): [Image domain of shape [B, Nx, Ny, 2]]
            k0 ([float32]): [K-space of the same shape]
            mask ([int32]): [K-Space mask of shape [B, Nx, Ny, 1]]

        Returns:
            [type]: [description]
        """     
        x = x.permute(0, 3, 1, 2).cuda()
        k0 = k0.permute(0, 3, 1, 2).cuda()
        mask = mask.permute(0, 3, 1, 2).cuda()
        result = self.perform(x, k0, mask)
        return result.permute(0, 2, 3, 1) # [B, H, W, 2]

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        # if x.dim() == 4: # input is 2D
        #     x    = x.permute(0, 2, 3, 1)
        #     k0   = k0.permute(0, 2, 3, 1)
        #     mask = mask.permute(0, 2, 3, 1)
        # elif x.dim() == 5: # input is 3D
        #     x    = x.permute(0, 4, 2, 3, 1)
        #     k0   = k0.permute(0, 4, 2, 3, 1)
        #     mask = mask.permute(0, 4, 2, 3, 1)

        # k = torch.fft.fft2(x, norm=self.normalized)
        k = fft(x)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        #x_res = torch.fft.ifft2(out, norm=self.normalized)
        x_res = ifft(out)

        # if x.dim() == 4:
        #     x_res = x_res.permute(0, 3, 1, 2)
        # elif x.dim() == 5:
        #     x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res
