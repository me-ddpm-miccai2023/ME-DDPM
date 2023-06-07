import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from lib.mri.scanner import Undersampler
from lib.diffusion.module import DCDiffusion
from lib.defaults import np_precision

class Model(pl.LightningModule):
    def __init__(
        self,
        acc_rate=4.0,
        no_central_lines=8,
        N=1000
    ):
        super().__init__()

        self.Undersampler = Undersampler(
            acc_rate=acc_rate,
            no_central_lines=no_central_lines,
            dtype=np_precision
        )
        self.Diffusion = DCDiffusion(K=N)
        self.N = N

        self.debug=False

    def convert_data(self, data):
        data["image"] = data["image"].cuda()
        return data

    def forward(self, x, u=None, v=None, meddpm=False, mask=None):
        x = self.convert_data(x)
        image = x["image"]  # [B, T, H, W]
        i_s = image.shape
        us = self.Undersampler(image[...,None], mask=mask)
        us["input_image"] = torch.reshape(us["input_image"], i_s) # [B, T, H, W]
        us["k_space"] = torch.reshape(us["k_space"], i_s) # [B, T, H, W]
        us["mask"] = torch.reshape(us["mask"], i_s) # [B, T, H, W]
        return self.Diffusion.infer(
            None,
            us["input_image"],
            us["k_space"],
            us["mask"],
            u=u,
            v=v,
            meddpm=meddpm
        ), us["input_image"], us["mask"]
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, prefix="val_")

    def training_step(self, x, batch_idx, prefix=''):
        x = self.convert_data(x)
        image = x["image"]
        us = self.Undersampler(image) 
        results = self.Diffusion.forward(
            image,
            us["input_image"],
            us["k_space"],
            us["mask"],
            None
        )
        loss = results["loss"]
        self.log(f"{prefix}train_loss", torch.sum(loss))
        self.log(f"{prefix}my_loss", torch.sum(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return torch.sum(loss)

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer