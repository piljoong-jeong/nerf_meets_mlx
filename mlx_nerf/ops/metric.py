import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

@mx.no_grad() # FIXME
def loss_to_PSNR(loss: mx.Tensor):

    return NotImplementedError

class MSE:
    def __call__(self, pred, gt):
        return mx.mean((pred - gt) ** 2)

class PSNR:
    def __call__(self, pred, gt):
        return 10 * mx.log10(1 / MSE()(pred, gt))
    
class SSIM:
    def __call__(self, pred, gt, w_size=11, size_average=True, full=False):
        
        # NOTE: set boundary values
        _max = 255 if mx.max(pred) > 128 else 1
        _min = -1 if mx.min(pred) < -0.5 else 0
        L = _max - _min
        c1 = ((k1 := 0.01) * L) ** 2
        c2 = ((k2 := 0.03) * L) ** 2

        _, channel, height, width = pred.size()
        window = self.create_window(w_size, channel)# TODO: find out equivalent of .to(pred.device)

    def create_window(self, w_size, channel):

        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window
    
    def gaussian(self, w_size, sigma):

        gaussian = mx.array([
            math.exp(-(x - w_size//2) ** 2) / float(2*sigma ** 2)
            for x in range(w_size)
        ])

        return gaussian / gaussian.sum()
    
class LPIPS:
    def __init__(self) -> None:
        import lpips
        self.model = lpips.LPIPS(net="vgg") # TODO: to mlx?

    def __call__(self, pred, gt, normalized=True):
        if normalized:
            pred = pred * 2.0 - 1.0
            gt = gt * 2.0 - 1.0
        error = self.model.forward(pred, gt)
        return mx.mean(error)