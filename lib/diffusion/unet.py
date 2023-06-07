
import pytorch_lightning as pl
import torch
from vendors.guided_diffusion.guided_diffusion.unet import UNetModel as UNet
from lib.defaults import image_size, use_fp16, precision, reduced_cap

CONDITIONED = True
class ForwardModel(pl.LightningModule):
    def __init__(
        self,
        architecture=None,
        conditioned_channels=4
    ):
        super().__init__()
        if architecture is None:
            self.forward_model = UNet(
                image_size=image_size,
                in_channels=conditioned_channels if CONDITIONED else 2,
                model_channels=16 if reduced_cap else 32, 
                out_channels=2,
                num_res_blocks=2,
                attention_resolutions=[16, 32],
                dropout=0,
                channel_mult=(1, 1, 2, 2, 4, 4),
                conv_resample=True,
                dims=3,
                num_classes=None,
                use_checkpoint=False,
                use_fp16=use_fp16,
                num_heads=4,
                num_head_channels=32 if reduced_cap else 64,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                resblock_updown=True,
                use_new_attention_order=False,).cuda()
        else:
            self.forward_model = architecture.cuda()

    def forward(self, x, timesteps, condition=None):
        # Expect [B, T, H, W, 1]
        assert(x.dtype in [torch.complex64, torch.complex128])
        x_ = torch.cat(
            [
                x.real.to(precision),
                x.imag.to(precision)
            ],
            dim=4
        )  # [B, T, H, W, 2]

        if (condition is not None) and CONDITIONED:
            condition_ = torch.cat(
                [
                    condition.real.to(precision),
                    condition.imag.to(precision)
                ],
                dim=4
            )  # [B, T, H, W, 2]
            x_ = torch.cat([x_, condition_], dim=4)

        x_ = torch.permute(x_, [0, 4, 1, 2, 3]) # [B, 2, T, H, W]
        y = self.forward_model(x_, timesteps)
        y = torch.permute(y, [0, 2, 3, 4, 1])

        y_real = y[..., 0:1]
        y_imag = y[..., 1:2]

        return torch.complex(
            y_real.to(torch.float32),
            y_imag.to(torch.float32)
        )
