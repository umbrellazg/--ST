import torch as th
import numpy as np


class NaiveFourierKANLayer(th.nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter(th.randn(2, outdim, inputdim, gridsize) /
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if (self.addbias):
            self.bias = th.nn.Parameter(th.zeros(1, outdim))

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = th.reshape(x, (-1, self.inputdim))
        # Starting at 1 because constant terms are in the bias
        k = th.reshape(th.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = th.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = th.cos(k * xrshp)
        s = th.sin(k * xrshp)
        y = th.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += th.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias
        # End fuse
        y = th.reshape(y, outshape)
        return y
