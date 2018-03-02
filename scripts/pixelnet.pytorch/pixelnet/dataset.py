import numpy as np
import torch.utils.data as data
import scipy.io as sio

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_mat(filename):
    return filename.endswith('.mat')

def make_dataset(dir, mode):
    images = []
    mats = []
    for root, _, fnames in sorted(os.walk(os.path.join(dir, 'images', mode))):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    for root, _, fnames in sorted(os.walk(os.path.join(dir, 'groundTruth', mode))):
        for fname in fnames:
            if is_mat(fname):
                path = os.path.join(root, fname)
                mats.append(path)
    images.sort()
    mats.sort()
    assert len(mats) == len(images), 'lists of mats and images not the same length'
    out = zip(images, mats)
    for i, (img_path, mat_path) in enumerate(out):
        img = os.path.split(img_path)[1]
        mat = os.path.split(mat_path)[1]
        assert img[0:img.find('.')] == mat[0:mat.find('.')], 'lists of mats and images not consistent {} != {}'.format(img, mat)
    return out


def pil_loader(path):
    return Image.open(path).convert('RGB')

def load_mat(path):
    mat = sio.loadmat(path)['groundTruth']
    n = mat.shape[1]
    out = np.zeros(mat[0][0][0][0][1].shape, dtype='f')
    for i in range(n):
        out += mat[0][i][0][0][1]
    # only use positive labels where a consensus (>=3 out of 5) of humans agreed
    return out >= 3


def default_loader(path):
    return pil_loader(path)


class BSDS500(data.Dataset):

    def __init__(self, root, transform=None, mode='train', loader=default_loader):
        imgs = make_dataset(root, mode)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path[0])
        target = load_mat(path[1])
        if self.transform is not None:
            img = self.transform(img)

        # return img, target
        # plt.imshow(target)
        # plt.pause(0.01)
        return img #, target

    def __len__(self):
        return len(self.imgs)



class HEDTransform(object):
    """
    pytorch implementation of the data augmentation scheme described in Holistically-Nested Edge Detection
    """
    def __init__(self, mean, std, crop_size):
        self.mean = np.array(mean, dtype='f')
        self.std = np.array(std, dtype='f')
        self.crop_size = crop_size

    def transform(self, img, target):
        return img, target


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

    target_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

    testset = BSDS500(root='/mnt/external/Data/BSR/BSDS500/data/',
                      mode='train',
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
            plt.imshow(toshow)
            plt.pause(0.03)
            pass


    # transform = HEDTransform(mean = [ 0.485, 0.456, 0.406 ],
    #                          std = [ 0.229, 0.224, 0.225 ],
    #                          crop_size=224)
    # mat = load_mat('/mnt/external/Data/BSR/BSDS500/data/groundTruth/train/2092.mat')
    # img = pil_loader('/mnt/external/Data/BSR/BSDS500/data/images/train/2092.jpg')
    #
    # plt.ion()
    # while 1:
    #     out = transform.transform(img, mat)
    #     plt.subplot(121)
    #     plt.imshow(out[0])
    #     plt.subplot(122)
    #     plt.imshow(out[1])
    #     plt.pause(0.03)
