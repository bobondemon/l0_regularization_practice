from types import SimpleNamespace
from functools import partial

import torch
from torch import nn

from sparsereg.model.basic_l0_blocks import L0Gate
from sparsereg.helper.helper import conv2d_full_and_l0_param_num, dense_full_and_l0_param_num

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn, fix_and_open_gate=True):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.hparams = SimpleNamespace(c_in=c_in, c_red=c_red, c_out=c_out, fix_and_open_gate=fix_and_open_gate)

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), nn.BatchNorm2d(c_out["1x1"]), act_fn()
        )
        self.conv_1x1_gate = L0Gate(c_out["1x1"], fix_and_open_gate=fix_and_open_gate)

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn(),
        )
        self.conv_3x3_gate = L0Gate(c_out["3x3"], fix_and_open_gate=fix_and_open_gate)

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn(),
        )
        self.conv_5x5_gate = L0Gate(c_out["5x5"], fix_and_open_gate=fix_and_open_gate)

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn(),
        )
        self.max_pool_gate = L0Gate(c_out["max"], fix_and_open_gate=fix_and_open_gate)

    def get_cout_l0(self):
        cout_l0 = 0
        for m in self.modules():
            if type(m) == L0Gate:
                cout_l0 += torch.sum(m.get_gate_mask_when_inferencing() > 0)
        return cout_l0

    def cal_full_and_l0_param_num(self, c_in_l0):
        # MANUALLY setting the hparam to calculate the sparsity
        c_in, c_red, c_out = self.hparams.c_in, self.hparams.c_red, self.hparams.c_out
        total_l0_and_full_param_num = torch.tensor([0, 0])
        # 1x1 convolution branch
        conv_1x1_l0 = torch.sum(self.conv_1x1_gate.get_gate_mask_when_inferencing() > 0)
        total_l0_and_full_param_num += conv2d_full_and_l0_param_num(c_in, c_out["1x1"], 1, c_in_l0, conv_1x1_l0)
        # 3x3 convolution branch
        total_l0_and_full_param_num += torch.tensor([c_in_l0 * c_red["3x3"], c_in * c_red["3x3"]])  # 1x1 conv
        conv_3x3_l0 = torch.sum(self.conv_3x3_gate.get_gate_mask_when_inferencing() > 0)
        total_l0_and_full_param_num += conv2d_full_and_l0_param_num(
            c_red["3x3"], c_out["3x3"], 3, c_red["3x3"], conv_3x3_l0
        )
        # 5x5 convolution branch
        total_l0_and_full_param_num += torch.tensor([c_in_l0 * c_red["5x5"], c_in * c_red["5x5"]])  # 1x1 conv
        conv_5x5_l0 = torch.sum(self.conv_5x5_gate.get_gate_mask_when_inferencing() > 0)
        total_l0_and_full_param_num += conv2d_full_and_l0_param_num(
            c_red["5x5"], c_out["5x5"], 5, c_red["5x5"], conv_5x5_l0
        )
        # Max-pool branch: No parameters
        return total_l0_and_full_param_num

    def forward(self, x):
        x_1x1 = self.conv_1x1_gate(self.conv_1x1(x))
        x_3x3 = self.conv_3x3_gate(self.conv_3x3(x))
        x_5x5 = self.conv_5x5_gate(self.conv_5x5(x))
        x_max = self.max_pool_gate(self.max_pool(x))
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, act_fn_name="relu", fix_and_open_gate=True, **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes, act_fn_name=act_fn_name, act_fn=act_fn_by_name[act_fn_name]
        )
        self.fix_and_open_gate = fix_and_open_gate
        self.inception_block = partial(InceptionBlock, fix_and_open_gate=fix_and_open_gate)
        self._create_network()
        self._init_params()

        self.l0_modules = [m for m in self.modules() if type(m) is L0Gate]

    def _create_network(self):
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), self.hparams.act_fn()
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            self.inception_block(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn,
            ),
            self.inception_block(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            self.inception_block(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            self.inception_block(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            self.inception_block(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            self.inception_block(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            self.inception_block(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            self.inception_block(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
        )
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, self.hparams.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x

    def reset_l0_parameters(self):
        if not self.fix_and_open_gate:
            for m in self.l0_modules:
                m.reset_parameters()
        else:
            print(
                f"[Warning]: Do nothing because `fix_and_open_gate` is True, which means the gate parameter is NOT learnable, and set as open"
            )

    def regularization(self):
        if self.fix_and_open_gate:
            # No L0 regularization
            return torch.tensor(0.0)
        # [TODO]: should implement l0 and/or l2
        reg = 0.0
        dim = 0
        for m in self.l0_modules:
            reg += m.regularization()
            dim += m.mask_dim

        # print(reg, dim)
        return reg / dim

    def cal_full_and_l0_param_num(self):
        # MANUALLY setting the hparam to calculate the sparsity
        total_l0_and_full_param_num = torch.tensor([0, 0])
        # self.input_net:
        total_l0_and_full_param_num += 3 * 64 * 3 * 3 + 64

        c_in_l0 = 64
        # Stacking inception blocks:
        for inception_block in self.inception_blocks:
            if type(inception_block) == InceptionBlock:
                total_l0_and_full_param_num += inception_block.cal_full_and_l0_param_num(c_in_l0)
                c_in_l0 = inception_block.get_cout_l0()

        # Mapping to classification output
        # The full dense layer matrix: (32+64+16+16)*self.hparams.num_classes
        total_l0_and_full_param_num += dense_full_and_l0_param_num(
            32 + 64 + 16 + 16, self.hparams.num_classes, c_in_l0, self.hparams.num_classes
        )
        return total_l0_and_full_param_num
