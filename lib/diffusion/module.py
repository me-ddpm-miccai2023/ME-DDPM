from argparse import ArgumentError
import pytorch_lightning as pl
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from typing import Union
from ..image.metrics import L2_loss
from ..mri.dc import DataConsistencyInKspace as DataConsistency
from .unet import ForwardModel
from ..mri.mri_math import fft2c as fft, ifft2c as ifft
from lib.defaults import debug
from ..motion.warp import execute_flow_warp
from lib.image.metrics import PSNR
import matplotlib.pyplot as plt


class DCDiffusion(nn.Module):
    def __init__(
        self,
        K: int,
        forward_model: Union[torch.nn.Module, pl.LightningModule] = ForwardModel,
        learn_beta: bool = False
    ):
        self.betas = None
        self.betas_tilde = None
        self.alphas = None
        self.alphas_tilde = None

        super(DCDiffusion, self).__init__()
        self.__init_betas(learn_beta, K)
        self.__init_alphas()
        self.forward_model = forward_model()

        self.generator = torch.Generator()
        self.generator.manual_seed(1117)
        self.DC = DataConsistency()
        self.K = K

        self.debug = debug

    def forward(self, image, us_image, k_space, mask, k):
        if k is None:
            B = image.shape[0]
            k = torch.rand(
                B,
            )
            k = (self.K * k).int().cuda()
            k = k + 1

        noisy_image, noise = self.noising(image, k)
        noise_pred = self.forward_model(noisy_image, k, condition=us_image)
        loss = L2_loss(noise, noise_pred)

        return {
            "noisy_image": noisy_image,
            "input_image": image,
            "k": k,
            "noise_pred": noise_pred,
            "loss": loss,
            "gt_noise": noise
        }

    def infer(
            self, noisy_image, us_image, k_space, mask, u=None, v=None,
            meddpm=False, gt=None):
        us_image = us_image.cuda()
        k_space = k_space.cuda()
        mask = mask.cuda()
        if meddpm is False:
            y = noisy_image
            for i in tqdm.tqdm(range(self.K)):
                y = self.infer_k(
                    y, us_image, k_space, mask, torch.from_numpy(
                        np.array([self.K - i])).cuda(),
                    )
                y = y.detach()
        else:
            u = u.cuda() 
            v = v.cuda()  
            y = noisy_image
            for i in tqdm.tqdm(range(self.K)):
                y = self.infer_k_dcmac(
                    y, us_image[0, :, ..., None], # [T, H, W, 1]
                    k_space[0, :, ..., None],
                    mask[0, :, ..., None],
                    torch.from_numpy(np.array([self.K - i])).cuda(),
                    u=u, v=v)
                y = y.detach()
        return y
    
    def infer_k(self, noisy_image, us_image, k_space, mask, k):
        assert(k >= 1 and k <= self.K)
        kk = k-1
        sigma_squared = self.betas_tilde[kk]
        sigma = torch.pow(sigma_squared, 0.5)

        if k == self.K:
            assert(noisy_image is None)
            noisy_image = torch.normal(0., 1., size=us_image.shape).cuda()
        if k > 1:
            z = torch.normal(0., 1., size=us_image.shape).cuda()
        else:
            assert(k == 1)
            z = 0.

        noise_pred = self.forward_model(noisy_image, k, condition=us_image)
        normalised_error_t1 = 1 - self.alphas[kk]
        normalised_error_t2 = torch.pow(1 - self.alphas_tilde[kk], 0.5)
        normalisation = normalised_error_t1/normalised_error_t2
        new_pred = (1/torch.pow(self.alphas[kk], 0.5)) * (
            noisy_image - (normalisation*noise_pred)
        )
        new_pred = new_pred + (sigma*z)
        for _ in range(2):
            t1_ = torch.pow(self.alphas_tilde[k-1], 0.5)
            new_pred = self.DC(new_pred, k_space*t1_, mask)
            noise_pred = self.forward_model(new_pred, k-1, condition=us_image)
            zz = torch.normal(0., 1., size=new_pred.shape).cuda()
            ep = 0.5*(0.1*sigma)**2
            epsq = torch.sqrt(2*ep) * zz
            new_pred = new_pred - ep*noise_pred
            new_pred = new_pred + epsq

        return new_pred

    def infer_k_dcmac(self, noisy_image, us_image, k_space, mask, k, u, v):
        assert(k >= 1 and k <= self.K)
        kk = k-1
        sigma_squared = self.betas_tilde[kk]
        sigma = torch.pow(sigma_squared, 0.5)
        if k == self.K:
            assert(noisy_image is None)
            noisy_image = torch.normal(0., 1., size=us_image.shape).cuda()
        if k > 1:
            z = torch.normal(0., 1., size=us_image.shape).cuda()
        else:
            assert(k == 1)
            z = 0.

        noise_pred = self.forward_model(noisy_image, k, condition=us_image)
        normalised_error_t1 = 1 - self.alphas[kk]
        normalised_error_t2 = torch.pow(1 - self.alphas_tilde[kk], 0.5)
        normalised_error = (noise_pred*normalised_error_t1)/normalised_error_t2
        noisy_image = noisy_image[..., 0]
        normalised_error = normalised_error[..., 0] 
        diff = noisy_image - normalised_error

        new_pred_t1 = (1/torch.pow(self.alphas[kk], 0.5)) * diff
        if k > 1:
            new_noise = sigma * z
            new_pred = new_pred_t1 + new_noise[..., 0]
        else:
            new_pred = new_pred_t1

        t1_ = torch.pow(self.alphas_tilde[k-1], 0.5)

        k_space_noise = k_space * t1_
        new_pred = self.DC(new_pred[..., None], k_space_noise, mask)[..., 0]

        Klimit = 50
        if k > Klimit:
            n_dcmac = 1
            for i in range(n_dcmac):
                new_pred = self.dcmac_step(
                    new_pred[..., None],
                    u, v, k_space * t1_)
                new_pred = new_pred[..., 0]
                new_pred = new_pred.detach()

                # DCLD
                noise_pred = self.forward_model(
                    new_pred, k-1, condition=us_image)
                zz = torch.normal(0., 1., size=new_pred.shape).cuda()
                ep = 0.5*(0.1*sigma)**2
                epsq = torch.sqrt(2*ep) * zz
                new_pred = new_pred - ep*noise_pred
                new_pred = new_pred + epsq

        if k > 1:
            for _ in range(1):
                t1_ = torch.pow(self.alphas_tilde[k-1], 0.5)
                new_pred = self.DC(new_pred, k_space*t1_, mask)
                noise_pred = self.forward_model(
                    new_pred, k-1, condition=us_image)
                zz = torch.normal(0., 1., size=new_pred.shape).cuda()
                ep = 0.5*(0.1*sigma)**2
                epsq = torch.sqrt(2*ep) * zz
                new_pred = new_pred - ep*noise_pred
                new_pred = new_pred + epsq
        else:
            new_pred = self.DC(new_pred, k_space, mask)
        return new_pred[..., None]

    def noising(self, image, ks):
        t1 = []
        t2 = []
        for k in ks:
            t1_ = torch.pow(self.alphas_tilde[k-1], 0.5)
            t2_ = torch.pow(1.-self.alphas_tilde[k-1], 0.5)

            t1.append(t1_)
            t2.append(t2_)
        epsilon = torch.normal(0., 1., size=image.shape).cuda()
        t1 = torch.FloatTensor(t1).cuda()
        t2 = torch.FloatTensor(t2).cuda()
        t1 = torch.reshape(t1, [-1, 1, 1, 1])
        t2 = torch.reshape(t2, [-1, 1, 1, 1])
        
        noisy_image = (t1*image) + (t2*epsilon)

        return noisy_image, epsilon

    def __init_betas(self, learn_beta, K):
        if learn_beta:
            self.betas = [_weight() for _ in range(K)]
        else:
            self.betas = torch.linspace(
                1.e-4,
                0.02,
                K
            )  # Ho et. al. (2020)

    def __init_alphas(self):
        self.alphas = [1-beta for beta in self.betas]
        self.alphas_tilde = [1]
        self.betas_tilde = []
        for alpha, beta in zip(self.alphas, self.betas):
            self.alphas_tilde.append(
                self.alphas_tilde[-1] * alpha
            )

            num = 1 - self.alphas_tilde[-2]
            den = 1 - self.alphas_tilde[-1]
            self.betas_tilde.append(
                (num/den) * beta
            )
        del(self.alphas_tilde[0])

        self.alphas_tilde = torch.FloatTensor(self.alphas_tilde).cuda()
        self.betas_tilde = torch.FloatTensor(self.betas_tilde).cuda()
        self.betas = torch.FloatTensor(self.betas).cuda()
        self.alphas = torch.FloatTensor(self.alphas).cuda()

    def dcmac_step(self, x, u, v, k_space, mask):
        u = u[:, None]
        v = v[:, None]
        x = x[:, None, ..., 0]
        x1 = execute_flow_warp(x, u, v)
        x1 = x1[:, 0, ..., None]
        x1 = torch.roll(x1, 1, 0)
        x1 = self.DC(x1, k_space[..., None], mask[..., None])
        return x1


def _weight():
    return torch.nn.Parameter(
        data=torch.Tensor(1), requires_grad=True)
