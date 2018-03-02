import numpy as np
import matplotlib.pyplot as plt
import os

filenames = ['h', 'dh', 'sdf', 'dsdfx', 'dsdfy', 'pb', 'pf'] #, 'Jwx', 'Jwy', 'Jwz', 'Jtx', 'Jty', 'Jtz'] #, 'active']
root_path = '../bin'

mats = {}
if __name__ == '__main__':
    for i, filename in enumerate(filenames):
        mats[filename] = np.loadtxt(os.path.join(root_path, filename))
        plt.figure(i)
        plt.imshow(mats[filename])
        plt.colorbar()
        plt.title(filename)

    plt.show()


