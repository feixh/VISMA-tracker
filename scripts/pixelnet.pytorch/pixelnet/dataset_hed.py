import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import os
import os.path

# def pil_loader(path):
#     return Image.open(path).convert('RGB')


class HED(data.Dataset):
    def __init__(self, root, transform=None, train=True, balance=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.pairs = []
        self.balance = balance

        if train:
            with open(os.path.join(root, 'train_pair.lst'), 'r') as f:
                lines = f.readlines()
                self.pairs = [tuple(line.rstrip().split(' ')) for line in lines]
        else:
            assert self.transform is not None, 'must set transform for testing!!!'
            with open(os.path.join(root, 'test.lst'), 'r') as f:
                lines = f.readlines()
                self.pairs = [line.rstrip() for line in lines]

    def __getitem__(self, index):
        if self.train:
            return self.__getitem_train(index)
        else:
            return self.__getitem_test(index)

    def __getitem_test(self, index):
        paths = self.pairs[index]
        # img = np.array(Image.open(os.path.join(self.root, paths[0])).convert('RGB'))
        img = Image.open(os.path.join(self.root, paths)).convert('RGB')
        img_t = self.transform(img)
        return img_t

    def __getitem_train(self, index):
        paths = self.pairs[index]
        # img = np.array(Image.open(os.path.join(self.root, paths[0])).convert('RGB'))
        img = np.array(Image.open(os.path.join(self.root, paths[0])).convert('RGB'), dtype='f') / 255.0
        target = np.array(Image.open(os.path.join(self.root, paths[1])))
        if target.ndim == 3:
            target = target[..., 0]
        elif target.ndim == 2:
            pass
        else:
            raise ValueError('invalid dimension of target')

        row, col = img.shape[0:2]

        # assert target is not None, 'target is None!!!'
        if row < col:
            sz = row
            img = img[:, col/2 -sz/2:col/2-sz/2 + sz, :]
            target = target[:, col/2-sz/2:col/2-sz/2+sz]
        else:
            sz = col
            sz = row
            img = img[row/2-sz/2:row/2-sz/2+sz, :, :]
            target = target[row/2-sz/2:row/2-sz/2+sz, :]

        img = (img - np.array([[[0.485, 0.465, 0.406]]])) / np.array([[[0.229, 0.224, 0.225]]])
        img = cv2.resize(img, (224, 224))
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)
        # FIXME: not sure how they extract binary edge map from these gray scale images ...
        target = cv2.resize(target, (224, 224)) > 0
        # target = cv2.resize(target, (224, 224))

        if not self.balance:
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(target[np.newaxis, ...].astype(np.float32))
        else:
            # return balancing weights
            weights = np.ones((224, 224), dtype=np.float32)
            pos_counter = target.sum()
            size = target.size
            weights[target] *= (size-pos_counter)/ float(size)
            weights[np.logical_not(target)] *= pos_counter / float(size)
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(target[np.newaxis, ...].astype(np.float32)), torch.from_numpy(weights[np.newaxis, ...].astype(np.float32))

    def __len__(self):
        return len(self.pairs)


def test_dataset_train():
    batch_size = 5

    # Data loading code
    transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    std = [ 0.229, 0.224, 0.225 ]),
    ])

    dataroot = '/mnt/external/Data/HED-BSDS'
    testset = HED(root= dataroot,
                  train=True,
                  transform=transform,
                  balance=True)

    test_loader = data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)

    plt.ion()
    for i, (img, target, weight) in enumerate(test_loader):
        for j in range(batch_size):
            toshow = img[j, ...].numpy()
            gt = target[j, 0, ...].numpy()
            w = weight[j, 0, ...].numpy()
            toshow = np.swapaxes(toshow, 0, 1)
            toshow = np.swapaxes(toshow, 1, 2)
            plt.clf()
            plt.subplot(131)
            plt.imshow(toshow)
            plt.subplot(132)
            plt.imshow(gt, cmap='hot')
            plt.subplot(133)
            plt.imshow(w, cmap='hot')
            plt.pause(0.03)
            pass

def restore_image(img):
    std = np.array([[[ 0.229,  0.224,  0.225]]])
    mean = np.array([[[0.485, 0.456, 0.406]]])
    img = img * std + mean
    return (img*255).astype(np.uint8)


if __name__ == '__main__':
    batch_size = 5

    # Data loading code
    transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    std = [ 0.229, 0.224, 0.225 ]),
    ])

    dataroot = '/mnt/external/Data/HED-BSDS'
    testset = HED(root= dataroot,
                  train=False,
                  transform=transform)

    test_loader = data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)

    plt.ion()
    for i, img in enumerate(test_loader):
        for j in range(batch_size):
            toshow = img[j, ...].numpy()
            toshow = np.swapaxes(toshow, 0, 1)
            toshow = np.swapaxes(toshow, 1, 2)
            plt.clf()
            plt.subplot(121)
            plt.imshow(toshow)
            plt.subplot(122)
            plt.imshow(restore_image(toshow))
            plt.pause(0.03)
            pass
