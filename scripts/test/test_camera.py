# test camera wrapper of myrender.so
from feh_render.camera import Camera

import trimesh
import numpy as np
from time import time
import matplotlib.pyplot as plt

from transforms3d import axangles

from scipy.ndimage import gaussian_filter
import cv2


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

    camera = Camera(480, 640)
    camera.set_camera(0.05, 10.0, [400, 400, 320, 240])
    camera.set_mesh(v, f)

    # Single render test -- warm up
    t1 = time()
    depth = camera.render_depth(model_pose)
    print('Warm up: Render depth takes {} seconds'.format(time()-t1))
    plt.imshow(depth, cmap='hot')
    plt.title('GL initialization and depth rendering')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # Single render test
    t1 = time()
    depth = camera.render_depth(model_pose)
    print('2nd render takes {} seconds'.format(time()-t1))
    plt.imshow(depth, cmap='hot')
    plt.title('Depth rendering without GL initialization')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    t1 = time()
    edge = cv2.Laplacian(depth.astype(np.uint8), cv2.CV_8U, ksize=5)
    print('OpenCV edge extraction takes {} seconds'.format(time()-t1))
    plt.imshow(edge, cmap='hot')
    plt.title('Edge extraction runs on depth map')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    evidence = edge.astype(np.uint8)
    camera.upload_evidence(evidence)

    t1 = time()
    edge = camera.render_edge(model_pose)
    # edge = cv2.Canny((depth*255).astype(np.uint8), 10, 50)
    print('OpenGL edge extraction takes {} seconds'.format(time()-t1))
    plt.imshow(edge, cmap='hot')
    plt.title('OpenGL edge extraction')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    t1 = time()
    gpu_likelihood = camera.likelihood(model_pose)
    t2 = time()
    print('OpenGL likelihood takes {} seconds'.format(t2-t1))
    cpu_likelihood = np.sum(np.minimum(edge / 255.0, evidence / 255.0))
    print('gpu likelihood={}; cpu likelihood={}'.format(gpu_likelihood, cpu_likelihood))
    assert np.abs(gpu_likelihood - cpu_likelihood) / np.abs(gpu_likelihood + cpu_likelihood) < 0.1, 'inconsistent likelihood'

    camera.max_pool_evidence(21)
    gpu_likelihood = camera.likelihood(model_pose)
    print('after max-pooling, gpu likelihood={}'.format(gpu_likelihood))

    total_time = 0
    plt.ion()
    num_iters = 72
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])
        rand_pose = np.hstack((R, T[:, np.newaxis])).astype('f')

        t1 = time()
        edge = camera.render_edge(rand_pose)
        t1 = time() - t1
        total_time += t1

        plt.clf()
        plt.imshow(edge, cmap='hot')
        plt.pause(0.001)
    print('@Given object pose OpenGL edge extraction average time={} ms'.format(1000*total_time/num_iters))

    total_time = 0
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])
        rand_pose = np.hstack((R, T[:, np.newaxis])).astype('f')

        t1 = time()
        depth = camera.render_depth(rand_pose)
        t1 = time() - t1
        total_time += t1

        plt.clf()
        plt.imshow(depth, cmap='hot')
        plt.pause(0.001)
    print('@Given object pose OpenGL depth rendering average time={} ms'.format(1000*total_time/num_iters))

    total_time = 0
    camera.use_intersection_kernel()    # actually not need to set, since intersection kernel is used by default
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])

        t1 = time()
        camera.likelihood(rand_pose)
        t1 = time() - t1
        total_time += t1
    print('@Given object pose OpenGL intersection kernel evaluation average time={} ms'.format(1000*total_time/num_iters))

    total_time = 0
    camera.use_cross_entropy()
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])

        t1 = time()
        camera.likelihood(rand_pose)
        t1 = time() - t1
        total_time += t1
    print('@Given object pose OpenGL intersection kernel evaluation average time={} ms'.format(1000*total_time/num_iters))

    plt.ioff()



