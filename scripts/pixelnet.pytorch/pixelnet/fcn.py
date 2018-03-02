import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from model import vgg16
from torchvision.models import vgg16

import glog

from time import time

from helper import msr_init

class FCN8(nn.Module):

    def __init__(self, num_classes):
        super(FCN8, self).__init__()

        glog.info('loading vgg ...')
        feats = list(vgg16(pretrained=True).features.children())

        # self.feats = nn.Sequential(*feats[0:9])
        # self.feat3 = nn.Sequential(*feats[10:16])
        # self.feat4 = nn.Sequential(*feats[17:23])
        # self.feat5 = nn.Sequential(*feats[24:30])

        self.feature = nn.ModuleList(
            [nn.Sequential(*feats[0:9]),
             nn.Sequential(*feats[10:16]),
             nn.Sequential(*feats[17:23]),
             nn.Sequential(*feats[24:30])
            ])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        # self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        # self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        # self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score = nn.ModuleList(
            [nn.Conv2d(256, num_classes, 1),
             nn.Conv2d(512, num_classes, 1),
             nn.Conv2d(4096, num_classes, 1)
            ])

        msr_init(self.fconn)
        msr_init(self.score)

    def forward(self, x):
        input_size = x.size()[2:]
        feats = self.feats[0](x)
        feats3 = self.feats[1](feats)
        feats4 = self.feats[2](feats3)
        feats5 = self.feats[3](feats4)
        feats_fconn = self.fconn(feats5)

        # feats = [feats, feats3, feats4, feats5, feats_fconn]

        score_feat3 = self.score[0](feats3)
        score_feat4 = self.score[1](feats4)
        score_fconn = self.score[2](feats_fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4
        score = F.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3

        return F.upsample_bilinear(score, input_size)


class FCN16(nn.Module):

    def __init__(self, num_classes):
        super(FCN16, self).__init__()

        feats = list(vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])


class FCN32(nn.Module):

    def __init__(self, num_classes):
        super(FCN32, self).__init__()

        glog.info('loading vgg ...')
        self.feats = vgg16(pretrained=True).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, num_classes, 1)
        msr_init(self.fconn)
        msr_init(self.score)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)
        out = F.sigmoid(score)
        return F.upsample_bilinear(out, x.size()[2:])


if __name__ == '__main__':
    net = FCN32(1)
    net.cuda()
    dummy_input = torch.rand(5, 3, 224, 224).cuda()
    t0 = time()
    out = net.forward(Variable(dummy_input))
    t1 = time()
    glog.info('elapsed {} sec'.format(t1-t0))
