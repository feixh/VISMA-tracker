from time import time
import os
import argparse
import seaborn

# google
import glog

# torch
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

# the model and criteria
from pixelnet import hypercolumn
from pixelnet import hypercolumn_sparse
from pixelnet import segnet

from pixelnet import dataset_hed as dataset
from pixelnet.helper import Colors

# logging system
from tensorboard_logger import configure, log_value

if __name__ == '__main__':
    ########################################
    # reload customized modules
    ########################################
    reload(dataset)
    reload(hypercolumn)
    reload(hypercolumn_sparse)
    reload(segnet)

    ########################################
    # arguments
    ########################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', default='/mnt/external/Data/HED-BSDS',
                        help='root directory of the data')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='size of each minibatch')
    parser.add_argument('--num-epoch', default=10, type=int,
                        help='number of epochs to train')
    parser.add_argument('--model', default='hc_base',
                        help='choose from: hc_base/hc_sparse/segnet')
    parser.add_argument('--resume-epoch', default=-1, type=int,
                        help='resume training from a certain epoch; '
                             'train from scratch if negative')
    parser.add_argument('--learning-rate', default=1e-3, type=float,
                        help='learning rate')
    # flags
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='not use cuda if on')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visualize if on')
    parser.add_argument('--use-SGD', action='store_true', default=False,
                        help='use SGD for optimization if on; default is ADAM')
    # model specific
    parser.add_argument('--classifier-grid', default=5, type=int,
                        help='devide the domain into N x N grid of classifiers')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='number of pixels to sample; only for sparse hypercolumn model')
    args = parser.parse_args()

    ########################################
    # setup log file
    ########################################
    if not os.path.exists(args.model):
        glog.warn('path {} not exists; creating one ...'.format(
            Colors.cyan(os.path.abspath(args.model))))
        os.makedirs(args.model)
    logfile = os.path.join(args.model, 'log.txt')

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

    train_set = dataset.HED(root=args.dataroot,
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
    # configure logging system
    ########################################
    # configure('runs/{}'.format(args.model))

    ########################################
    # setup optimizer
    ########################################
    if args.model == 'hc_base' or args.model == 'hc_sparse':
        encoder = net.feature
        decoder = net.hc
    else:
        encoder = net.encoder
        decoder = net.decoder

    if args.use_SGD:
        glog.info('training with SGD')
        optimizer = torch.optim.SGD(
            [{'params': encoder.parameters(), 'lr': 1e-5, 'tag': 'encoder'},
             {'params': decoder.parameters(), 'lr': args.learning_rate, 'tag': 'decoder'}],
            momentum=0.9
        )
    else:
        glog.info('training with ADAM')
        optimizer = torch.optim.Adam(
            [{'params': encoder.parameters(), 'lr': 1e-5, 'tag': 'encoder'},
             {'params': decoder.parameters(), 'lr': args.learning_rate, 'tag': 'decoder'}],
            betas=(0.9, 0.9)
        )

    # load parameters from previous iteration
    if args.resume_epoch >= 0:
        tmp = torch.load(os.path.join(args.model, 'model_epoch{:04d}.pth'.format(args.resume_epoch)))
        net.load_state_dict(tmp['net'])
        optimizer.load_state_dict(tmp['optimizer'])
        start_epoch = tmp['epoch'] + 1
        glog.check_eq(args.resume_epoch, tmp['epoch'], 'epoch number inconsistent')
        glog.info('model trained up to epoch {:4d} with loss={:0.4f} loaded'.format(start_epoch - 1, tmp['loss']))
    else:
        start_epoch = 0
        best_acc = 0.0
        glog.info('train a model from scratch')
        if os.path.exists(logfile):
            glog.warn(Colors.red('train from scratch!!! removing existing logs!!!'))
            os.remove(logfile)  # clean up the log file
    net.train()  # training mode

    ########################################
    # the loop
    ########################################
    batch_size = args.batch_size
    num_batch = len(train_loader)
    seaborn.set_style('darkgrid')
    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        ###############################################
        # adjust learning rate
        ###############################################
        lr = args.learning_rate * (0.1 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            if param_group['tag'] == 'decoder':
                param_group['lr'] = lr
            elif param_group['tag'] == 'encoder':
                glog.info('constant lr for encoder (vgg) ...')
        glog.info(Colors.cyan('learning rate of decoder is {} after {} epoch\033[0m'.format(lr, epoch)))

        ###############################################
        # training loop
        ###############################################
        total_loss = 0.0
        net.train()
        glog.info('training, epoch {:6d}'.format(epoch))
        tic = time()
        total_iter = 0
        for i, (images, targets, weights) in enumerate(train_loader):
            if args.no_cuda:
                img, target = Variable(images), Variable(targets, requires_grad=False)
            else:
                img, target = Variable(images.cuda()), Variable(targets.cuda(), requires_grad=False)
            # weight = Variable(weights.cuda(), requires_grad=False)

            optimizer.zero_grad()

            # FIXME: form and insert weights to balance the data
            if args.model == 'hc_sparse':
                # forward pass
                out, samples = net.forward(img)
                # sample targets
                sampled_targets = Variable(hypercolumn_sparse.sample_from(targets, samples), requires_grad=False)
                sampled_weights = hypercolumn_sparse.sample_from(weights, samples)
                if not args.no_cuda:
                    sampled_targets, sampled_weights = sampled_targets.cuda(), sampled_weights.cuda()
                loss = F.binary_cross_entropy(input=out.view(1, -1), target=sampled_targets.view(1, -1),
                                              weight=sampled_weights.view(1, -1))
            else:
                # forward pass
                out = net.forward(img)
                # sample targets
                loss = F.binary_cross_entropy(input=out.view(1, -1), target=target.view(1, -1),
                                              weight=weights.cuda().view(1, -1))

            loss.backward()
            optimizer.step()

            total_iter = len(train_loader) * epoch + i
            total_loss += loss.data[0]

            if (i+1) % 20 == 0:
                glog.info('epoch{:4} -- {:8}/{:8}'.format(epoch, i, len(train_loader)))
                glog.info(Colors.green('train')
                          + ' running loss={:0.4f}; average running time={:0.4f} sec/batch'.format(
                    total_loss / (i + 1),
                    (time() - tic) / (i + 1)))
                # log_value('train_loss', loss.data[0], total_iter)

        toc = time()

        # logging info
        ss = 'epoch={:4d}, loss={:0.4f}'.format(epoch, total_loss / (i + 1))
        with open(logfile, 'a') as f:
            f.write(ss + '\n')

        # save model
        glog.info(Colors.red(ss))
        data_to_save = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': total_loss / (i + 1),
            'epoch': epoch
        }
        torch.save(data_to_save, os.path.join(args.model, 'model_epoch{:04d}.pth'.format(epoch)))

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
