from vlslam_tools import vlslam_pb2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from time import time

from vlslam_tools.utils import load_edgemap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/mnt/external/tmp/swivel_chair',
                        help='dataset root')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        filenames = [args.path]
    else:
        filenames = glob.glob(os.path.join(args.path, '*.edge'))
        filenames.sort(key=lambda x: float(os.path.basename(x)[:-5]))

    plt.ion()
    for i, filename in enumerate(filenames):
        edgemap = load_edgemap(filename)
        plt.clf()
        plt.imshow(edgemap, cmap='hot')
        plt.pause(0.01)
