from types import SimpleNamespace
import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
import torch.nn.functional as F


class L0Gate(nn.Module):
    def __init__(
        self, mask_dim, droprate_init=0.5, temperature=2.0 / 3.0, weight_decay=1.0, lamba=1.0, fix_and_open_gate=True
    ):
        # `fix_and_open_gate = True` means by using a very large `qz_loga` the mask equals to 1 (gate open) with a very high probability
        # `droprate_init`: the lower the value is, the higher the probability of mask=1.
        #                  (Good for training with a pre-trained model, i.e. init with open gates)
        super(L0Gate, self).__init__()

        self.limit_a, self.limit_b, self.epsilon = -0.1, 1.1, 1e-6
        self.mask_dim = mask_dim  # mask dimension
        self.droprate_init = droprate_init
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.fix_and_open_gate = fix_and_open_gate

        # qz_loga indicates logit_ratio: log prob1/prob2, so we set qz_loga very large to make prob1 becomes nearly 1
        if self.fix_and_open_gate:
            self.register_buffer("qz_loga", 999 * torch.ones(self.mask_dim))
        else:
            self.qz_loga = nn.parameter.Parameter(torch.Tensor(self.mask_dim))
            self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        # print(self.qz_loga.shape)

    def reset_parameters(self):
        # used when loading from a pre-trained model (fix_and_open_gate=False), and we want qz_loga becomes learnable
        if not self.fix_and_open_gate:
            self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        else:
            print(
                f"[Warning]: Do nothing because `fix_and_open_gate` is True, which means the gate parameter is NOT learnable, and set as open"
            )

    def get_uniform_rv(self, mask_shape):
        """Uniform random numbers for the concrete distribution"""
        uniform_rv = torch.Tensor(mask_shape).uniform_(self.epsilon, 1 - self.epsilon)
        return uniform_rv

    def quantile_concrete(self, uniform_rv):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        uniform_rv = uniform_rv.to(self.qz_loga.device)
        binary_concrete_rv = torch.sigmoid(
            (torch.log(uniform_rv) - torch.log(1 - uniform_rv) + self.qz_loga) / self.temperature
        )
        stretched_binary_concrete_rv = binary_concrete_rv * (self.limit_b - self.limit_a) + self.limit_a
        return stretched_binary_concrete_rv

    def sample_hard_concrete_rv(self, mask_shape, is_train=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if is_train:
            uniform_rv = self.get_uniform_rv(mask_shape)
            stretched_binary_concrete_rv = self.quantile_concrete(uniform_rv)
            hard_binary_concrete_rv = F.hardtanh(stretched_binary_concrete_rv, min_val=0, max_val=1)
            return hard_binary_concrete_rv
        else:  # mode
            # [TODO]: Check following codes are used when inferencing ...
            # mask_shape[0] is batch_size which should be the same during inferencing
            pi = torch.sigmoid(self.qz_loga).view(1, mask_shape[1])
            return F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def forward(self, input_):
        # input_: (n, c, h, w) for conv2d's output, or (n, dim_z) for dense layer's output
        mask_shape = input_.shape[:2]
        ndim = input_.ndim
        assert ndim >= 2, "[Error]: only support input_.ndim>=2"
        assert mask_shape[1] == self.mask_dim, "[Error]: the mask dim should be equal to input_.shape[1]"
        mask = self.sample_hard_concrete_rv(mask_shape, is_train=self.training)  # (n, c)
        for _ in range(ndim - 2):
            mask = torch.unsqueeze(mask, -1)
        return input_ * mask

    def cdf_sctretched_concrete(self, x_bar):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # cdf_sctretched_concrete(0) = Pr(mask=0)
        # A mistake I made: using torch.Tensor(0) will produce a tensor with size=0 (scalar)
        x_bar = torch.tensor(x_bar).to(self.qz_loga.device)
        x = (x_bar - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(x) - math.log(1 - x)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=self.epsilon, max=1 - self.epsilon)

    def mask_is_non_zero_probability(self):
        # output, Pr(mask=\=0), has the same shape as self.qz_loga
        return 1 - self.cdf_sctretched_concrete(0)

    def regularization(self):
        # Is it WRONG with the signed value in the author's codes:
        # https://github.com/AMLab-Amsterdam/L0_regularization/blob/39a5fe68062c9b8540dba732339c1f5def451f1b/l0_layers.py#L69
        return torch.sum(self.lamba * self.mask_is_non_zero_probability())
