from vlslam_tools import vlslam_pb2
from im2hog_demo import demo as im2hog_demo
JointEmbedding = im2hog_demo.JointEmbedding
import argparse
import glob
import os
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import cv2
import skimage
import copy

import im2hog_demo.model_selection as model_selection
shapenet_to_scanned = model_selection.shapenet_to_scanned

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/mnt/external/tmp/swivel_chair',
                        help='dataset root')
    parser.add_argument('--dummy-shapeid', default="",
                        help='use the string as shapeid if the string is non-empty')
    parser.add_argument('--no-visualization', default=False, action='store_true',
                        help='suppress visualization')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        filenames=[args.path]
    else:
        filenames = glob.glob(os.path.join(args.path, '*.bbox'))
        filenames.sort(key=lambda x: float(os.path.basename(x)[:-5]))

    if not args.no_visualization:
        plt.ion()

    joint_embed = JointEmbedding('im2hog_demo/hog_sammon_03001627.npz', 'im2hog_demo/checkpoint/5.ckpt')
    # fig1 = plt.figure(1)
    # ax_debug = fig1.add_subplot(111)
    fig2 = plt.figure(2)
    ax_main = fig2.add_subplot(111)

    for i, filename in enumerate(filenames):
        print('{}/{}'.format(i, len(filenames)))
        image_file = filename[:-5] + '.png'
        # image = np.array(PIL.Image.open(image_file))
        image = np.array(skimage.io.imread(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxlist = vlslam_pb2.BoundingBoxList()
        with open(filename, 'rb') as fid:
            bboxlist.ParseFromString(fid.read())

        for bbox in bboxlist.bounding_boxes:
            pt1 = int(bbox.top_left_x), int(bbox.top_left_y)
            pt2 = int(bbox.bottom_right_x), int(bbox.bottom_right_y)

            # if not bbox.HasField("shape_id"):
            # FIXME: run shape retrieval
            if len(args.dummy_shapeid) > 0:
                print('dummy shapeid')
                bbox.shape_id = args.dummy_shapeid
            else:
                print('retrieval')
                patch = image[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
                # ax_debug.imshow(patch)
                bbox.shape_id = shapenet_to_scanned[joint_embed.retrieve(patch, 1)[0]]

            shape_id = bbox.shape_id

            if not args.no_visualization:
                cv2.rectangle(image, pt1, pt2, (255, 205, 51), 2)
                cv2.putText(image, '%s: %.3f' % (bbox.class_name, bbox.scores[0]),
                            (pt1[0], pt1[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0, (255, 0, 0), thickness=2)
                cv2.putText(image, 'shapeid: %s' % (shape_id),
                            (pt1[0], pt1[1] + 30),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0, (255, 255, 0), thickness=2)

        with open(filename, 'wb') as fid:
            fid.write(bboxlist.SerializeToString())

        if not args.no_visualization:
            ax_main.cla()
            ax_main.imshow(image)
            plt.pause(0.001)




