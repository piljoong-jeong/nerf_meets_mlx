"""### encoding

"""
from abc import abstractmethod

import mlx.core as mx
import mlx.nn as nn


class Encoding(nn.Module):
    def __init__(self, in_dim: int) -> None:
        self.in_dim = in_dim

    @abstractmethod
    def forward(self, in_array: mx.array):
        raise NotImplementedError
    
    @abstractmethod
    def get_out_dim(self):
        raise NotImplementedError