import matplotlib.pyplot as plt
import numpy as np
import glob
import PIL.Image
import os

if __name__ == '__main__':
    data_root = '../dump/'
    particle_files = glob.glob(os.path.join(data_root, '*.txt'))
    particle_files.sort()
    image_files = glob.glob(os.path.join(data_root, '*.png'))
    image_files.sort()
    plt.ion()
    fig = plt.figure(1, figsize=(24, 8))
    ax_x = fig.add_subplot(231)
    ax_y = fig.add_subplot(232)
    ax_z = fig.add_subplot(233)
    ax_t = fig.add_subplot(234)
    ax_w = fig.add_subplot(235)
    ax_i = fig.add_subplot(236)
    axes = [ax_x, ax_y, ax_z, ax_t, ax_w, ax_i]
    for frame_id, (particle_file, image_file) in enumerate(zip(particle_files, image_files)):
        particles = np.loadtxt(particle_file)
        image = np.array(PIL.Image.open(image_file))
        for i in range(5):
            ax = axes[i]
            ax.cla()
            # ax.plot(particles[:, i], '.')
            if i == 3:
                ax.hist(particles[:, i] / 3.14, bins=50)
            else:
                ax.hist(particles[:, i], bins=50)

        ax_i.cla()
        ax_i.imshow(image)

        plt.pause(0.01)
        plt.savefig(os.path.join(data_root, 'viz_%06d.png' % frame_id))




