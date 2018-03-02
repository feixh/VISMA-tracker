import numpy as np
import os

from torch import nn
from graphviz import Digraph
from torch.autograd import Variable
import torch.utils.data as data

import torch
from functools import reduce
from operator import mul
import numpy as np

import math

import glog


def msr_init(net):
    """
    MSR style initialization
    :param net:
    :return:
    """
    glog.info('initialization with MSR approach')
    try:
        for layer in net:
            if type(layer) == nn.Conv2d:
                n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
                layer.weight.data.normal_(0, math.sqrt(2./n))
                layer.bias.data.zero_()
            elif type(layer) == nn.BatchNorm2d:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()
            elif type(layer) == nn.Linear:
                layer.bias.data.zero_()
    except TypeError:
        glog.warn('input argument is not iterable ... try to treat it as a single module')
        if type(net) == nn.Conv2d:
            n = net.kernel_size[0]*net.kernel_size[1]*net.out_channels
            net.weight.data.normal_(0, math.sqrt(2./n))
            net.bias.data.zero_()
        elif type(net) == nn.BatchNorm2d:
            net.weight.data.fill_(1)
            net.bias.data.zero_()
        elif type(net) == nn.Linear:
            net.bias.data.zero_()


class Colors:
    @staticmethod
    def red(x): return '\033[91m' + x + '\033[0m'
    @staticmethod
    def green(x): return '\033[92m' + x + '\033[0m'
    @staticmethod
    def blue(x): return '\033[94m' + x + '\033[0m'
    @staticmethod
    def cyan(x): return '\033[96m' + x + '\033[0m'
    @staticmethod
    def yellow(x): return '\033[93m' + x + '\033[0m'
    @staticmethod
    def magenta(x): return '\033[95m' + x + '\033[0m'

    Red = '\033[91m'
    Green = '\033[92m'
    Blue = '\033[94m'
    Cyan = '\033[96m'
    White = '\033[97m'
    Yellow = '\033[93m'
    Magenta = '\033[95m'
    Grey = '\033[90m'
    Black = '\033[90m'
    Default = '\033[99m'
    End = '\033[0m'
    Bold = '\033[1m'
    Underline = '\033[4m'

#####################################################
# Advanced Indexing taken from
# https://gist.github.com/fmassa/f8158d1dfd25a8047c2c668a44ff57f4
#####################################################

def _linear_index(sizes, indices):
    indices = [i.view(-1) for i in indices]
    linear_idx = indices[0].new(indices[0].numel()).zero_()
    stride = 1
    for i, idx in enumerate(indices[::-1], 1):
        linear_idx += stride*idx
        stride *= sizes[-i]
    return linear_idx


def advanced_indexing(tensor, index):
    if isinstance(index, tuple):
        adv_loc = []
        for i, el in enumerate(index):
            if isinstance(el, torch.LongTensor):
                adv_loc.append((i, el))
        if len(adv_loc) < 2:
            return tensor[index]

        # check that number of elements in each indexing array is the same
        len_array = [i.numel() for _, i in adv_loc]
        #assert len_array.count(len_array[0]) == len(len_array)

        idx = [i for i,_ in adv_loc]
        sizes = [tensor.size(i) for i in idx]
        new_size = [tensor.size(i) for i in range(tensor.dim()) if i not in idx]
        new_size_final = [tensor.size(i) for i in range(tensor.dim()) if i not in idx]

        start_idx = idx[0]
        # if there is a space between the indexes
        if idx[-1] - idx[0] + 1 != len(idx):
            permute = idx + [i for i in range(tensor.dim()) if i not in idx]
            tensor = tensor.permute(*permute).contiguous()
            start_idx = 0

        lin_idx = _linear_index(sizes, [i for _, i in adv_loc])
        reduc_size = reduce(mul, sizes)
        new_size.insert(start_idx, reduc_size)
        new_size_final[start_idx:start_idx] = list(adv_loc[0][1].size())

        tensor = tensor.view(*new_size)
        tensor = tensor.index_select(start_idx, lin_idx)
        tensor = tensor.view(new_size_final)

        return tensor

    else:
        return tensor[index]


def compare_numpy(t, idxs):
    r = advanced_indexing(t, idxs).numpy()
    np_idxs = tuple(i.tolist() if isinstance(i, torch.LongTensor) else i for i in idxs)
    r2 = t.numpy()[np_idxs]
    assert np.allclose(r, r2)
    assert r.shape == r2.shape


if __name__ == '__main__':
    t = torch.rand(3,3,3)
    idx1 = torch.LongTensor([0,1])
    idx2 = torch.LongTensor([1,1])

    compare_numpy(t, (idx1, slice(0,3), idx2))
    compare_numpy(t, (slice(0,3), idx1, idx2))


    t = torch.rand(10,20,30,40,50)
    idx_dim = (2,3,4)
    idx1 = torch.LongTensor(*idx_dim).random_(0, 20-1)
    idx2 = torch.LongTensor(*idx_dim).random_(0, 30-1)

    compare_numpy(t, (slice(0, None), idx1, idx2))

    idx2 = torch.LongTensor(*idx_dim).random_(0, 40-1)

    compare_numpy(t, (slice(0, None), idx1, slice(0, None), idx2))

    idx3 = torch.LongTensor(*idx_dim).random_(0, 50-1)
    compare_numpy(t, (slice(0, None), idx1, slice(0, None), idx2, idx3))
