import torch
from pixelnet import segnet
import glog
from pixelnet.helper import Colors
import numpy as np

model_path = '/local/feixh/workspace/simplerender/scripts/pixelnet.pytorch/segnet_epoch0009.pth'

stats = {'mean': [0.485, 0.456, 0.406],
         'std': [0.229, 0.224, 0.225]}


def load_model():
    """
    Load SegNet trained for edge detection.
    :return: network and stats of dataset
    """
    net = segnet.SegNet(n_classes=1,
                        in_channels=3,
                        is_unpooling=True,
                        pretrained=False)
    net.cuda()
    tmp = torch.load(model_path)
    net.load_state_dict(tmp['net'])
    start_epoch = tmp['epoch']+1
    loss = tmp['loss']
    glog.info('model trained up to epoch {} with loss {:0.4f} loaded'.format(start_epoch-1, loss))
    net.eval()  # test mode

    return net


def convert_image_to_tensor(img):
    """
    Convert an RGB image to a tensor of NCHW form.
    :param img: input RGB image
    :return: NCHW tensor
    """
    return torch.from_numpy(img.swapaxes(1, 2).swapaxes(0, 1)[np.newaxis, ...])


def normalize_image(img):
    """
    Normalize an image such that intensity ~ N(0, 1)
    :param img: input image
    :return: normalized image
    """
    if img.max() <= 1:
        glog.warn(Colors.yellow('intensity value already in [0, 1]?'))
    else:
        normalized_image = img.astype('f') / 255

    for i in range(3):
        normalized_image[..., i] = (normalized_image[..., i] - stats['mean'][i]) / stats['std'][i]
    return normalized_image
