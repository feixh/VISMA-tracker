# Test the render pool
# Size of render pool is 20
# Pick a renderer from the pool and render depth image using it
from feh_render.camera import Camera

import trimesh
import numpy as np
from time import time
import matplotlib.pyplot as plt

from transforms3d import axangles


if __name__ == '__main__':
    path = './resources/chair.obj'
    mesh = trimesh.load(path)
    v, f = np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.float64)
    # v[:, 1] = v[:, 1] - v[:, 1].min() + 0.08
    v = v - np.median(v, axis=0)
    v[:, 1] = -v[:, 1]

    R = axangles.axangle2mat(axis=[0, 1, 0], angle=np.pi/6)
    T = np.array([0, 0, 1])
    model_pose = np.hstack((R, T[:, np.newaxis])).astype('f')

    Camera.clear_camera_pool()

    plt.ion()
    for i in range(20):
        camera = Camera(480, 640)
        camera.set_camera(0.05, 10.0, [400, 400, 320, 240])
        camera.set_mesh(v, f)
        print('This is camera %i' % camera.renderer_index)
        depth = camera.render_depth(model_pose)
        plt.imshow(depth, cmap='hot')
        plt.pause(0.01)
