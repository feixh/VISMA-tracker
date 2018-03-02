import numpy as np
from time import time
import os
import argparse
import matplotlib.pyplot as plt
import seaborn

# google
import glog

# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

# the model and criteria
import fcn as model
import dataset_hed as dataset
from helper import Colors

if __name__ == '__main__':
    ########################################
    # reload customized modules
    ########################################
    reload(model)
    reload(dataset)

    ########################################
    # arguments
    ########################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--num-epoch', default=10, type=int)
    # parser.add_argument('--data-path', default='triplets_points_0505')
    parser.add_argument('--model-path', default='fcn32')
    # parser.add_argument('--train-id', default=0, type=int)
    parser.add_argument('--resume', default=-1, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)  # visualize testing on training data?
    parser.add_argument('--use-SGD', action='store_true', default=False)
    args = parser.parse_args()

    ########################################
    # setup log file
    ########################################
    if not os.path.exists(args.model_path):
        glog.warn('path {} not exists; creating one ...'.format(Colors.cyan(args.model_path)))
        os.makedirs(args.model_path)
    logfile = os.path.join(args.model_path, 'log.txt')

    ########################################
    # construct the net
    ########################################
    net = model.FCN32(num_classes=1)
    if not args.no_cuda:
        net.cuda()

    if args.resume >= 0:
        tmp = torch.load(os.path.join(args.model_path, 'model_epoch{:04d}.pth'.format(args.resume)))
        net.load_state_dict(tmp['net'])
        start_epoch = tmp['epoch'] + 1
        glog.check_eq(args.resume, tmp['epoch'], 'epoch number inconsistent')
        glog.info('model trained up to epoch {:4d} with loss={:0.4f} loaded'.format(start_epoch - 1, tmp['loss']))
    else:
        start_epoch = 0
        best_acc = 0.0
        glog.info('train a model from scratch')
        if os.path.exists(logfile):
            os.remove(logfile)  # clean up the log file
    net.train()  # training mode

    ########################################
    # load data and make it multiple of batch size
    ########################################
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataroot = '/mnt/external/Data/HED-BSDS'
    train_set = dataset.HED(root=dataroot,
                            train=True,
                            transform=transform,
                            balance=True)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)

    # uncomment the following block for debugging
    # plt.ion()
    # for i, (img, target) in enumerate(train_loader):
    #     for j in range(args.batch_size):
    #         toshow = img[j, ...].numpy()
    #         gt = target[j, 0, ...].numpy()
    #         toshow = np.swapaxes(toshow, 0, 1)
    #         toshow = np.swapaxes(toshow, 1, 2)
    #         plt.clf()
    #         plt.subplot(121)
    #         plt.imshow(toshow)
    #         plt.subplot(122)
    #         plt.imshow(gt, cmap='hot')
    #         plt.pause(0.03)
    #         pass

    ########################################
    # setup optimizer
    ########################################
    if args.use_SGD:
        glog.info('training with SGD')
        optimizer = torch.optim.SGD(
            [{'params': net.feats.parameters(), 'lr': 1e-5, 'tag': 'vgg'},
             {'params': net.fconn.parameters(), 'lr': args.learning_rate, 'tag': 'hc'},
             {'params': net.score.parameters(), 'lr': args.learning_rate, 'tag': 'hc'}],
            momentum=0.9
        )
    else:
        glog.info('training with ADAM')
        optimizer = torch.optim.Adam(
            [{'params': net.feats.parameters(), 'lr': 1e-5, 'tag': 'vgg'},
             {'params': net.fconn.parameters(), 'lr': args.learning_rate, 'tag': 'hc'},
             {'params': net.score.parameters(), 'lr': args.learning_rate, 'tag': 'hc'}],
            betas=(0.9, 0.9)
        )

    ########################################
    # the loop
    ########################################
    batch_size = args.batch_size
    num_batch = len(train_loader)
    seaborn.set_style('darkgrid')
    for epoch in xrange(start_epoch, start_epoch + args.num_epoch):
        ###############################################
        # adjust learning rate
        ###############################################
        lr = args.learning_rate * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            if param_group['tag'] == 'hc':
                param_group['lr'] = lr
            elif param_group['tag'] == 'vgg':
                glog.info('constant lr for vgg ...')
        glog.info(Colors.cyan('learning rate of hc is {} after {} epoch\033[0m'.format(lr, epoch)))

        ###############################################
        # training loop
        ###############################################
        total_loss = 0.0
        net.train()
        glog.info('training, epoch {:6d}'.format(epoch))
        tic = time()
        for i, (images, targets, weights) in enumerate(train_loader):
            img = Variable(images.cuda())
            target = Variable(targets.cuda(), requires_grad=False)
            # weight = Variable(weights.cuda(), requires_grad=False)

            optimizer.zero_grad()
            out = net.forward(img)

            # FIXME: form and insert weights to balance the data
            loss = F.binary_cross_entropy(input=out.view(1, -1), target=target.view(1, -1),
                                          weight=weights.cuda().view(1, -1))

            total_loss += loss.data[0]

            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                glog.info('{:8}/{:8}'.format(i, len(train_loader)))
                glog.info(Colors.green('train')
                          + ' running loss={:0.4f}; average running time={:0.4f} sec/batch'.format(
                    total_loss / (i + 1),
                    (time() - tic) / (i + 1)))
        toc = time()

        # logging info
        ss = 'epoch={:4d}, loss={:0.4f}'.format(epoch, total_loss / (i + 1))
        with open(logfile, 'a') as f:
            f.write(ss + '\n')

        # save model
        glog.info(Colors.red(ss))
        tosave = {
            'net': net.state_dict(),
            'loss': total_loss / (i + 1),
            'epoch': epoch
        }
        torch.save(tosave, os.path.join(args.model_path, 'model_epoch{:04d}.pth'.format(epoch)))

        # ###############################################
        # # testing loop
        # ###############################################
        # net.eval()
        # glog.info('testing ...')
        # plt.ion()
        # for i, (images, targets) in enumerate(train_loader):
        #     img = Variable(images.cuda(), requires_grad=False)
        #     target = Variable(targets.cuda(), requires_grad=False)
        #
        #     out = net.forward(img)
        #
        #     for j in range(batch_size):
        #         toshow = images[j, ...].numpy()
        #         gt = targets[j, 0, ...].numpy()
        #         pred = out[j, 0, ...].data.cpu().numpy()
        #         toshow = np.swapaxes(toshow, 0, 1)
        #         toshow = np.swapaxes(toshow, 1, 2)
        #         plt.clf()
        #         plt.subplot(131)
        #         plt.imshow(toshow)
        #         plt.subplot(132)
        #         plt.imshow(gt, cmap='hot')
        #         plt.subplot(133)
        #         plt.imshow(pred, cmap='hot')
        #         plt.pause(0.03)
        #         pass
