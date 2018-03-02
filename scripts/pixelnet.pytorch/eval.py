import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

# google
import glog

# torch
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms


# the model and criteria
from pixelnet import hypercolumn
from pixelnet import hypercolumn_sparse
from pixelnet import segnet
from pixelnet import dataset_hed as dataset
from pixelnet.helper import Colors

if __name__ == '__main__':
    ########################################
    # reload customized modules
    ########################################
    reload(hypercolumn)
    reload(hypercolumn_sparse)
    reload(segnet)
    reload(dataset)

    ########################################
    # arguments
    ########################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/mnt/external/Data/HED-BSDS',
                        help='root directory of the data')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--resume-epoch', default=0, type=int)
    parser.add_argument('--model', default='segnet')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--test-set', action='store_true', default=False,
                        help='run model on test set if on; otherwise on training set')
    # model specific
    parser.add_argument('--classifier-grid', default=5, type=int,
                        help='devide the domain into N x N grid of classifiers')
    args = parser.parse_args()

    ########################################
    # construct the net
    ########################################
    if args.model == 'hc_base':
        net = hypercolumn.Base(K=args.classifier_grid,
                               pretrained=(args.resume_epoch < 0))
    elif args.model == 'hc_sparse':
        net = hypercolumn_sparse.Sparse(K=args.classifier_grid,
                                        pretrained=(args.resume_epoch < 0),
                                        num_samples=args.num_samples)
    elif args.model == 'segnet':
        net = segnet.SegNet(n_classes=1,
                            in_channels=3,
                            is_unpooling=True,
                            pretrained=(args.resume_epoch < 0))

    if not args.no_cuda:
        net.cuda()

    if args.resume_epoch >= 0:
        tmp = torch.load(os.path.join(args.model, 'model_epoch{:04d}.pth'.format(args.resume_epoch)))
        net.load_state_dict(tmp['net'])
        start_epoch = tmp['epoch']+1
        glog.check_eq(args.resume_epoch, tmp['epoch'], 'epoch number inconsistent')
        glog.info('model trained up to epoch {:4d} with loss={:0.4f} loaded'.format(start_epoch-1, tmp['loss']))
    else:
        raise ValueError('resume must be a non-negative integer!!')

    ########################################
    # load data and make it multiple of batch size
    ########################################
    transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_set = dataset.HED(root=args.dataroot,
                           train=(not args.test_set),
                           transform=transform)

    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1)

    ########################################
    # the loop
    ########################################
    batch_size = args.batch_size
    num_batch = len(test_loader)
    seaborn.set_style('darkgrid')

    ###############################################
    # testing loop
    ###############################################
    net.eval()
    glog.info('testing ...')
    plt.ion()

    fig = plt.figure()
    if args.test_set:
        ax_img = fig.add_subplot(121)
        ax_pred = fig.add_subplot(122)
    else:
        ax_img = fig.add_subplot(131)
        ax_gt = fig.add_subplot(132)
        ax_pred = fig.add_subplot(133)

    for i, datum in enumerate(test_loader):
        if not args.test_set:
            images, targets = datum
            if not args.no_cuda:
                img, target = Variable(images.cuda(), requires_grad=False), Variable(targets.cuda(), requires_grad=False)
            else:
                img, target = Variable(images, requires_grad=False), Variable(targets, requires_grad=False)
        else:
            images = datum
            if not args.no_cuda:
                img = Variable(images.cuda(), requires_grad=False)
            else:
                img = Variable(images, requires_grad=False)

        # TODO: make sparse hypercolumn predict entire image in inference mode
        out = net.forward(img)

        for j in range(batch_size):
            to_show = images[j, ...].numpy()
            pred = out[j, 0, ...].data.cpu().numpy()
            to_show = np.swapaxes(to_show, 0, 1)
            to_show = np.swapaxes(to_show, 1, 2)
            to_show = dataset.restore_image(to_show)

            ax_img.cla()
            ax_img.imshow(to_show)
            ax_img.axis('off')

            if not args.test_set:
                gt = targets[j, 0, ...].numpy()
                ax_gt.cla()
                ax_gt.imshow(gt, cmap='hot')
                ax_gt.axis('off')

            ax_pred.cla()
            im = ax_pred.imshow(pred, cmap='hot')
            ax_pred.axis('off')
            divider = make_axes_locatable(ax_pred)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            plt.pause(0.03)

