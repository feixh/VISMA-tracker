# This is for extracting edge map(s) for a single image file or a directory of dataset images.
import numpy as np
from torch.autograd import Variable
from network_utils import load_model, convert_image_to_tensor, normalize_image
import PIL.Image
import os
import argparse
import glob
import scipy.io as sio

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "georgia"

from vlslam_tools import vlslam_pb2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./resources/chair_overlay.png',
                        help='image path or a directory of dataset images')
    parser.add_argument('--save-edgemap', default=False, action='store_true',
                        help='save edge maps returned by CNN')
    parser.add_argument('--save-figure', default=False, action='store_true',
                        help='save figures')
    parser.add_argument('--save-as-mat', default=False, action='store_true',
                        help='save edge map as .mat file')
    parser.add_argument('--save-as-npy', default=False, action='store_true',
                        help='save edge map as .npy file')
    args = parser.parse_args()

    net = load_model()

    fig = plt.figure(1)
    ax_image = fig.add_subplot(121)
    ax_edge = fig.add_subplot(122)

    if os.path.isfile(args.path):
        filenames = [args.path]
        save_path = os.path.join(args.path, '..')
    elif os.path.isdir(args.path):
        filenames = glob.glob(args.path + '/*.png')
        try:
            filenames.sort(key=lambda x: float(os.path.basename(x)[:-4]))
        except ValueError:
            filenames.sort()
        save_path = args.path

    plt.ion()
    if args.save_as_mat:
        edgemap_mat = {}

    for i, filename in enumerate(filenames):
        print('{}/{}'.format(i, len(filenames)))
        image = np.array(PIL.Image.open(filename))
        var_img = Variable(convert_image_to_tensor(normalize_image(image)).cuda())
        out = net(var_img)
        # edge_map = build_pyramid(out[0, 0, ...].data.cpu().numpy(), scale_factor=2, levels=args.pyramid_level)[-1]
        # edge_map = gaussian(out[0, 0, ...].data.cpu().numpy(), sigma=args.blur_sigma)
        edge_map = out[0, 0, ...].data.cpu().numpy()
        if args.save_edgemap:
            # np.save(os.path.join(save_path, os.path.basename(filename)+'.edge'), edge_map)
            edge_msg = vlslam_pb2.EdgeMap()
            edge_msg.description = 'SegNet-based Edge Detection: {}'.format(filename)
            edge_msg.rows, edge_msg.cols = edge_map.shape
            for f in edge_map.ravel():
                edge_msg.data.append(float(f))
            with open(os.path.join(save_path, os.path.basename(filename)[:-4] + '.edge'), 'wb') as fid:
                fid.write(edge_msg.SerializeToString())

        if args.save_as_mat:
            edgemap_mat[os.path.basename(filename).split('.')[0]] = edge_map

        if args.save_as_npy:
            np.save(os.path.join(save_path, os.path.basename(filename)[:-4]), edge_map)

        # ax_image.cla()
        # ax_image.imshow(image)
        # ax_image.axis('off')
        # ax_image.set_title('Image#{:04d}'.format(i))
        #
        # ax_edge.cla()
        # ax_edge.imshow(edge_map, cmap='hot')
        # ax_edge.axis('off')
        # ax_edge.set_title('Boundary')

        # if args.save_figure:
        #     plt.savefig(os.path.join(args.save_path, os.path.basename(filename) + '.edgemap.png'),
        #                 transparent=True)
        # plt.pause(0.001)

    if args.save_as_mat:
        sio.savemat('edgemap.mat', edgemap_mat)



