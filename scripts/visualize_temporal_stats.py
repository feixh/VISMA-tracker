import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='input log file', type=str)
    args = parser.parse_args()
    data = np.loadtxt(args.file)

    print("std={}".format(np.std(data, axis=0)))
    print("mean={}".format(np.mean(data, axis=0)))

    plt.subplot(221)
    plt.plot(data[:, 0])
    plt.subplot(222)
    plt.plot(data[:, 1])
    plt.subplot(223)
    plt.plot(data[:, 2])
    plt.subplot(224)
    plt.plot(data[:, 3])
    plt.show()
