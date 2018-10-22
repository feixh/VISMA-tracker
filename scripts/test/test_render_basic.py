# test wrappers of myrender.so
import trimesh
import numpy as np
from time import time
import matplotlib.pyplot as plt

from feh_render.simplerender import render_depth, render_depth_at, render_edge, \
    render_edge_at, upload_evidence, likelihood, likelihood_at, set_mesh, set_camera, \
    initialize_renderer

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

    vertices = v.copy().astype('f')
    R = axangles.axangle2mat(axis=[0, 1, 0], angle=np.pi/6)
    T = np.array([0, 0, 1])
    model_pose = np.hstack((R, T[:, np.newaxis])).astype('f')
    v = v.dot(R.T) + T


    v = v.astype(np.float32)
    f = f.astype(np.int32)
    calibration = np.array([400, 400, 240, 320], dtype=np.float32)
    imgsize = np.array([480, 640], dtype=np.int32)
    depth = np.zeros((480, 640), dtype=np.uint8)
    edge = np.zeros((480, 640), dtype=np.uint8)

    # Single render test
    t1 = time()
    render_depth(v, f, calibration, imgsize, depth)
    print('render takes {} seconds'.format(time()-t1))
    plt.imshow(depth, cmap='hot')
    plt.title('GL initialization and depth rendering')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # Single render test
    t1 = time()
    render_depth(v, f, calibration, imgsize, depth)
    print('2nd render takes {} seconds'.format(time()-t1))
    plt.imshow(depth, cmap='hot')
    plt.title('Depth rendering without GL initialization')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # Single render test
    rand_pose = np.eye(3, 4).astype('f')
    t1 = time()
    render_depth_at(rand_pose, depth)
    print('@pose OpenGL depth render takes {} seconds'.format(time()-t1))
    plt.imshow(depth, cmap='hot')
    plt.title('Depth rendering without passing vertices and faces')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    t1 = time()
    edge = cv2.Laplacian(depth.astype(np.uint8), cv2.CV_8U, ksize=5)
    # edge = cv2.Canny((depth*255).astype(np.uint8), 10, 50)
    print('OpenCV edge extraction takes {} seconds'.format(time()-t1))
    plt.imshow(edge, cmap='hot')
    plt.title('Edge extraction runs on depth map')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    evidence = edge.astype(np.uint8)
    upload_evidence(evidence)

    t1 = time()
    render_edge(v, f, calibration, imgsize, edge)
    # edge = cv2.Canny((depth*255).astype(np.uint8), 10, 50)
    print('OpenGL edge extraction takes {} seconds'.format(time()-t1))
    plt.imshow(edge, cmap='hot')
    plt.title('OpenGL edge extraction')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    t1 = time()
    gpu_likelihood = likelihood(v, f)
    t2 = time()
    print('OpenGL likelihood takes {} seconds'.format(t2-t1))
    cpu_likelihood = np.sum(np.minimum(edge / 255.0, evidence / 255.0))
    print gpu_likelihood, cpu_likelihood

    # test likelihood at function
    set_mesh(vertices, f)
    gpu_likelihhod2 = likelihood_at(model_pose)
    # this should be the same as feeding in the vertices and faces after transformation
    print('Use transformed mesh with identity pose likelihood={}\nUse original mesh with object pose likelihood={}'.format(
        gpu_likelihood, gpu_likelihhod2))
    assert gpu_likelihhod2 == gpu_likelihood, "Inconsistent likelihood"

    total_time = 0
    plt.ion()
    num_iters = 72
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])
        rand_pose = np.hstack((R, T[:, np.newaxis])).astype('f')

        t1 = time()
        render_edge_at(rand_pose, edge)
        t1 = time() - t1
        total_time += t1

        plt.clf()
        plt.imshow(edge, cmap='hot')
        plt.pause(0.001)
    print('@Given object pose OpenGL edge extraction average time={} seconds'.format(total_time/num_iters))

    total_time = 0
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])
        rand_pose = np.hstack((R, T[:, np.newaxis])).astype('f')

        t1 = time()
        render_depth_at(rand_pose, depth)
        t1 = time() - t1
        total_time += t1

        plt.clf()
        plt.imshow(depth, cmap='hot')
        plt.pause(0.001)
    print('@Given object pose OpenGL depth rendering average time={} seconds'.format(total_time/num_iters))

    total_time = 0
    for i in range(num_iters):
        R = axangles.axangle2mat(axis=[0, 1, 0], angle=i*2*np.pi/num_iters)
        T = np.array([0, 0, 1.0])

        t1 = time()
        likelihood_at(rand_pose)
        t1 = time() - t1
        total_time += t1
    print('@Given object pose OpenGL likelihood average time={} seconds'.format(total_time/num_iters))

    plt.ioff()
