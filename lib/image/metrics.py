import torch
from functools import reduce
import operator

def L2_loss(ref, img, norm=False):
    s = ref.shape
    ss = len(s)
    loss = torch.sum(
            torch.add(
                torch.pow((ref-img).real, 2.),
                torch.pow((ref-img).imag, 2.)
            ),
        dim=list(range(1,ss,1)))

    if norm is True:
        loss_norm = torch.sum(
                torch.add(
                    torch.pow((ref).real, 2.),
                    torch.pow((ref).imag, 2.)
                ),
            dim=list(range(1,ss,1)))
        loss = loss/torch.sum(loss_norm)
    return loss

def RNMSE(ref, img):
    s = ref.shape
    ss = len(s)
    diff = torch.sum(torch.pow(ref-img, 2))
    diff = diff / torch.sum(torch.pow(ref, 2))
    return torch.pow(diff, 0.5)

def prod(inp):
    return reduce(operator.mul, inp, 1)

def PSNR(ref, img, max_val=1.0, use_complex=False):
    '''
    First dim is batch
    '''
    if use_complex is True:
        raise NotImplemented("Only works for absolute")
    ref_ = torch.abs(ref)
    img_ = torch.abs(img)
    diff = ref_ - img_
    axes = list(torch.range(1, len(ref_.shape)-1))
    axes = [x.int() for x in axes]
    N = prod(list(ref_.shape[1::]))
    N = torch.Tensor([N]).to(ref.device)
    mse = torch.sum(torch.pow(diff, 2), dim=axes)*(1/N)
    psnr = -10.*torch.log10(mse)
    return psnr
