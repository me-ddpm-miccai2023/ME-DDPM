import torch
from .compressed_sensing import cartesian_mask
from .mri_math import fft2c as fft, ifft2c as ifft
import numpy as np

class Undersampler(object):
    def __init__(
        self,
        acc_rate=None,
        no_central_lines=4,
        dtype=np.float32
    ):
        self.acc_rate = acc_rate
        self.no_central_lines = no_central_lines
        self.dtype = dtype

    def __call__(
        self,
        image,
        acc_rate=None,
        no_central_lines=None,
        mask=None
    ):
        acc_rate_ = self._get_acc_rate(acc_rate)
        no_central_lines_ = self._get_no_central_lines(no_central_lines)
        mask_shape = image.size()[:-1]
        if mask is None:
            mask_ = cartesian_mask(
                mask_shape,
                acc_rate_,
                no_central_lines=no_central_lines_,
                centred=True
            ) # Produced on CPU using fast numpy implementation
        else:
            print("Using provided mask in scanner.py!")
            mask_ = mask
        mask_ = mask_.astype(self.dtype)

        mask_ = torch.tensor(mask_).cuda()
        mask = mask_.unsqueeze(0) # [1, B, T, H, W]
        
        image_ = image.permute([4,0,1,2,3]) # [C, B, T, H, W]
        kspace = fft(image_)
        masked_kspace = kspace * mask

        us_image = ifft(masked_kspace) # [C, B, T, H, W]
        us_image = us_image.permute([1, 2, 3, 4, 0]) # [B, T, H, W, C]
        masked_kspace_ = masked_kspace.permute([1, 2, 3, 4, 0])
        
        mask = mask.squeeze(0) # [B, T, H, W]
        mask = mask.unsqueeze(4) # [B, T, H, W, 1]

        return {
            "input_image": us_image,
            "mask": mask,
            "k_space": masked_kspace_
        }

    def _get_acc_rate(self, acc_rate):
        if acc_rate is None:
            if self.acc_rate is None:
                raise AttributeError('Please set `acc_rate`')
            else:
                return self.acc_rate
        else:
            return acc_rate

    
    def _get_no_central_lines(self, no_central_lines):
        if no_central_lines is None:
            if self.no_central_lines is None:
                raise AttributeError('Please set `no_central_lines`')
            else:
                return self.no_central_lines
        else:
            return no_central_lines
        