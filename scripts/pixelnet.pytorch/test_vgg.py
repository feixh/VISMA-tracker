import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

import torch.utils.data
from torch.autograd import Variable

import torchvision.transforms as transforms

import dataset
import model

from imagenet_helper import label2text

def restore_image(img):
    std = np.array([[[ 0.229,  0.224,  0.225]]])
    mean = np.array([[[0.485, 0.456, 0.406]]])
    img = img * std + mean
    return (img*255).astype(np.uint8)

if __name__ == '__main__':
    reload(dataset)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/mnt/external/Data/BSR/BSDS500/data/images')
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--num-thread', default=4, type=int)

    args = parser.parse_args()

    vgg = model.vgg16(pretrained=True)
    # vgg.cuda()
    #
    # # Data loading code
    # transform = transforms.Compose([
    # transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    # std = [ 0.229, 0.224, 0.225 ]),
    # ])
    #
    # testdir = os.path.join(args.data, 'test')
    # test_set = dataset.BSDS500(testdir, transform)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_thread)
    #
    # plt.ion()
    # for i, img in enumerate(test_loader):
    #     out = vgg.forward(Variable(img.cuda()))
    #     _, idx = torch.max(out.data.cpu(), dim=1)
    #
    #     for j in range(args.batch_size):
    #         text = label2text[idx[j][0]]
    #         toshow = img[j, ...].numpy()
    #         toshow = np.swapaxes(toshow, 0, 1)
    #         toshow = np.swapaxes(toshow, 1, 2)
    #         plt.imshow(restore_image(toshow))
    #         plt.title(text)
    #         plt.pause(0.03)
    #         raw_input()
