import torch


def conv2d_full_and_l0_param_num(cin, cout, ksize, cin_l0, cout_l0, with_bias=True, with_bn=True):
    # calculate sparsity of convolution kernel
    ksize = torch.tensor(ksize)
    assert ksize.numel() <= 2, f"[Error]: ksize should be int or (int, int)"
    if ksize.numel() == 1:
        ksize = (ksize, ksize)

    l0_param_num = cin_l0 * ksize[0] * ksize[1] * cout_l0
    full_param_num = cin * ksize[0] * ksize[1] * cout
    if with_bias:
        l0_param_num += cout_l0
        full_param_num += cout
    # considering bn layer: weight/bias + running mean/var
    if with_bn:
        l0_param_num += cout_l0 * 4
        full_param_num += cout * 4
    return torch.tensor([l0_param_num, full_param_num])


def dense_full_and_l0_param_num(din, dout, din_l0, dout_l0, with_bias=True):
    # calculate sparsity of dense layer
    l0_param_num = din_l0 * dout_l0
    full_param_num = din * dout
    if with_bias:
        l0_param_num += dout_l0
        full_param_num += dout
    return torch.tensor([l0_param_num, full_param_num])
