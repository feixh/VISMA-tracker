###############################################
# edge detection based on hypercolumn features and **pixel sampling**
# author: Xiaohan Fei
# reference1: Hypercolumns for Object Segmentation and Fine-grained Localization
# link: https://arxiv.org/abs/1411.5752
# reference2: PixelNet: Representation of the pixels, by the pixels, and for the pixels.
# link: http://www.cs.cmu.edu/~aayushb/pixelNet/pixelnet.pdf
###############################################
from hypercolumn import *


class Sparse(Base):
    def __init__(self, K=5, pretrained=False, num_samples=2000):
        super(Sparse, self).__init__(K, pretrained)
        # number of pixel samples
        self.num_samples = num_samples
        # size of the feature map which goes into classifiers
        self.hc_input_size = [224, 112, 56, 28, 14]

    def __sample(self, batch_size):
        """
        :return: each column is a sample
        """
        return np.random.randint(low=0, high=self.N, size=(batch_size, 2, self.num_samples))

    def forward(self, x):
        if self.training:
            # glog.info('train called')
            return self.__forward(x)
        else:
            # glog.info('test called')
            return super(Sparse, self).forward(x)

    def __forward(self, x):
        batch_size = x.size(0)
        # ignore the actual batch size, i.e. use the same set of pixles
        samples = self.__sample(batch_size=1)
        if self.use_cuda:
            out = Variable(torch.zeros(batch_size, self.K*self.K, 1, self.num_samples).cuda())
        else:
            out = Variable(torch.zeros(x.size(0), self.K*self.K, 1, self.num_samples))

        for i in range(5):
            x = self.feature[i](x)
            channels = x.size(1)

            # flat view of feature map
            xv = x.view(batch_size, channels, 1, -1)

            # compute indices needed by on-demand computation
            idx, w = compute_nearby_locations(self.N, self.hc_input_size[i], samples)
            # print idx.shape
            # print w.shape

            if self.use_cuda:
                w = Variable(torch.from_numpy(w[0, ...]), requires_grad=False).cuda()
                idx = torch.from_numpy(idx[0, ...]).cuda()
            else:
                w = Variable(torch.from_numpy(w[0, ...]), requires_grad=False)
                idx = torch.from_numpy(idx[0, ...])

            # bilinear interpolation
            xw = torch.sum(xv[:, :, [0], idx] * w, dim=2)
            # print xv[:, :, [0], idx].size()
            # print w.size()
            # print xw.size()

            # aggregate scores of linear classifiers
            out += self.hc[i](xw.view(batch_size, channels, 1, -1))

        out = F.sigmoid(out)
        # sample from interpolation grid
        sampled_grid = sample_from(self.interpolation_grid, samples)
        return torch.sum(sampled_grid * out, dim=1), samples


def sample_from(src, samples):
    """
    sample from src tensor/variable according to samples
    :parm src: 4-d tensor with form NCHW
    :param samples: 1 x 2 x num_sample numpy array
    :return: sampled quantity
    """
    assert samples.ndim == 3, 'invalid dimension of samples'
    num_sample = samples.shape[2]
    batch_size = src.size(0)
    tmp = []
    for i in range(num_sample):
        x, y = samples[0, :, i]
        tmp.append(src[:, :, x, y].contiguous().view(batch_size, -1, 1, 1))
    return torch.cat(tmp, dim=3)


def compute_nearby_locations(original_size, target_size, samples):
    """
    compute the four nearby locations for feature interpolation
    :param original_size: size of the original image from which we sample pixels
    :param target_size: size of the target feature map from which we build the hypercolumn feature for the sampled pixel
    :param samples: sampled pixel locations at the original image, each column is a sample
    :return: batch_size x (4x2 locations) x num_samples
    """
    batch_size = samples.shape[0]
    assert batch_size == 1, 'cannot only handle batch size of 1!!!'
    num_samples = samples.shape[-1]
    anchors = np.zeros((batch_size, 8, num_samples), dtype='f')
    o = original_size // 2
    ot = target_size // 2
    s = target_size / float(original_size)
    # FIXME: sampling is buggy, conditioned on target_size is odd or even number,
    # should handle differently
    # targets
    t = s*(samples-o) + ot
    x, y = t[:, [0], :], t[:, [1], :]
    # lower bound of x
    xl = np.maximum(np.floor(x), 0)
    # upper bound of x
    xu = np.minimum(xl+1, target_size-1)
    # lower bound of y
    yl = np.maximum(np.floor(y), 0)
    # upper bound of y
    yu = np.minimum(yl+1, target_size-1)
    # distance to xl
    dxl = x - xl
    dxu = xu - x
    dyl = y - yl
    dyu = yu - y
    # order: (xl, yl), (xl, yu), (xu, yl), (xu, yu)
    dx = dxl + dxu
    dx_nonzero = np.logical_not(dx < np.finfo('f').eps)
    alpha = np.ones((batch_size, 1, num_samples), dtype='f')*0.5
    alpha[dx_nonzero] = 1-dxl[dx_nonzero] / dx[dx_nonzero]

    dy = dyl + dyu
    dy_nonzero = np.logical_not(dy < np.finfo('f').eps)
    beta = np.ones((batch_size, 1, num_samples), dtype='f')*0.5
    beta[dy_nonzero] = 1-dyl[dy_nonzero] / dy[dy_nonzero]

    xlyl = (xl*target_size+yl).astype('i')
    xlyu = (xl*target_size+yu).astype('i')
    xuyl = (xu*target_size+yl).astype('i')
    xuyu = (xu*target_size+yu).astype('i')
    weights = np.concatenate((alpha*beta, alpha*(1-beta), (1-alpha)*beta, (1-alpha)*(1-beta)), axis=1).astype('f')
    return np.concatenate((xlyl.astype(np.int64),
                           xlyu.astype(np.int64),
                           xuyl.astype(np.int64),
                           xuyu.astype(np.int64)), axis=1), weights


if __name__ == '__main__':
    hc = Sparse()
    # hc.eval()
    dummy_input = torch.rand(5, 3, 224, 224)
    # out = hc.forward(Variable(dummy_input))
    t0 = time()
    out, samples = hc.forward(Variable(dummy_input))
    t1 = time()
    glog.info('elapsed {} sec'.format(t1-t0))

    # sample from the target
    dummy_target = Variable((torch.rand(5, 1, 224, 224) > 0.5).float())
    tmp = sample_from(dummy_target, samples).squeeze(dim=2)

    # print out.size()
    # print tmp.size()
    loss = F.binary_cross_entropy(out, tmp)
