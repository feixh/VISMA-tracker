import numpy as np
import argparse
import trimesh
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objfile', default='resources/chair.obj',
                        help='path to the .obj file')
    args = parser.parse_args()
    mesh = trimesh.load(args.objfile)
    vertices = np.array(mesh.vertices).astype('f')
    faces = np.array(mesh.faces).astype('i')
    tag = os.path.basename(args.objfile).split('.')[0]
    np.savetxt(tag + '_vertices.txt', vertices, '%f')
    np.savetxt(tag + '_faces.txt', faces, '%d')
