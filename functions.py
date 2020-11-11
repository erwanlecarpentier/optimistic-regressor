import numpy as np
import torch


class Quadratic(object):

    def forward(self, x):
        return x * x


class CSin(object):

    def forward(self, x):
        return (0.5*torch.sin(x * 6.28) + 0.5) * (0.5*torch.sin(x * 4.0 * 6.28) + 0.5)
