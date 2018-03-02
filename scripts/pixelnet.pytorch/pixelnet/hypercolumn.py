###############################################
# edge detection based on hypercolumn features
# author: Xiaohan Fei
# reference: Hypercolumns for Object Segmentation and Fine-grained Localization
# link: https://arxiv.org/abs/1411.5752
###############################################
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.models import vgg16

import glog

from time import time

from helper import msr_init


class Base(nn.Module):
    def __init__(self, K=5, pretrained=False):
        super(Base, self).__init__()
        glog.info('loading vgg ...')
        # load vgg model
        feats = list(vgg16(pretrained=pretrained).features.children())
        self.feature = nn.ModuleList(
            [nn.Sequential(*feats[0:4]),
             nn.Sequential(*feats[4:9]),
             nn.Sequential(*feats[9:16]),
             nn.Sequential(*feats[16:23]),
             nn.Sequential(*feats[23:30])
             ])
        # size of the classifier grid
        self.K = K
        # classifier as convolutional layers on hypercolumn
        # kernel size is a parameter we can tune: > 1 means consider nearby features
        self.hc = nn.ModuleList(
            [nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64, K * K, kernel_size=(1, 1))),
                nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, K * K, kernel_size=(1, 1))),
                nn.Sequential(
                    nn.BatchNorm2d(256),
                    nn.Conv2d(256, K * K, kernel_size=(1, 1))),
                nn.Sequential(
                    nn.BatchNorm2d(512),
                    nn.Conv2d(512, K * K, kernel_size=(1, 1))),
                nn.Sequential(
                    nn.BatchNorm2d(512),
                    nn.Conv2d(512, K * K, kernel_size=(1, 1)))
            ])
        # image size
        self.N = 224
        self.upsample = nn.Upsample(size=(self.N, self.N), mode='bilinear')
        self.interpolation_grid = Variable(torch.from_numpy(make_interpolation_grid(K=self.K, N=self.N, gamma=0.1)))
        msr_init(self.hc)
        self.use_cuda = False

    def forward(self, x):
        # glog.info('forward in base')
        if x.is_cuda:
            out = Variable(torch.zeros(x.size(0), self.K * self.K, self.N, self.N)).cuda()
            # self.interpolation_grid.repeat()
        else:
            out = Variable(torch.zeros(x.size(0), self.K * self.K, self.N, self.N))
        for i in range(5):
            x = self.feature[i](x)
            out += self.upsample(self.hc[i](x))
        out = F.sigmoid(out)
        # classifier interpolation and reduce along dimension 1, which has KxK classifiers
        return torch.sum(self.interpolation_grid * out, dim=1)

    def cuda(self):
        """
        reload cuda function
        """
        super(Base, self).cuda()
        self.use_cuda = True
        self.interpolation_grid = self.interpolation_grid.cuda()


def make_interpolation_grid(K=5, N=224, gamma=0.1):
    """
    make the K^2 x N x N grid for classifier interpolation
    :param K: size of classifier grid, we have KxK classifiers over the NxN image
    :param N: image size
    :param batch_size:
    :param gamma: intuively sigma^2 in Gaussian distribution
    """
    a = np.zeros((K * K, N, N), dtype='f')
    sz = N / float(K)
    for row in range(N):
        for col in range(N):
            for ki in range(K):
                for kj in range(K):
                    k = ki * K + kj
                    i, j = int(row / sz), int(col / sz)
                    a[k, row, col] = np.exp(-gamma * ((ki - i) ** 2 + (kj - j) ** 2))
    a = a / a.sum(axis=0)
    return a[np.newaxis, ...]


def test_grid():
    a = make_interpolation_grid(5, 224, 0.1)
    for i in range(224):
        for j in range(224):
            plt.clf()
            plt.imshow(a[:, i, j].reshape((5, 5)), cmap='hot')
            plt.title('({},{})'.format(i, j))
            plt.colorbar()
            plt.pause(0.03)
            raw_input()


if __name__ == '__main__':
    hc = Base()
    hc.cuda()
    dummy_input = torch.rand(5, 3, 224, 224).cuda()
    t0 = time()
    out = hc.forward(Variable(dummy_input))
    t1 = time()
    glog.info('elapsed {} sec'.format(t1 - t0))
