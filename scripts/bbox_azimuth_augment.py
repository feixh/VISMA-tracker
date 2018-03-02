from viewpoint_estimation.main import estimate_azimuth, load_viewpoint_estimator
from vlslam_tools import vlslam_pb2
import argparse
import glob
import os
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/mnt/external/tmp/swivel_chair',
                        help='dataset root')
    args = parser.parse_args()

    net=load_viewpoint_estimator()

    if os.path.isfile(args.path):
        filenames=[args.path]
    else:
        filenames = glob.glob(os.path.join(args.path, '*.bbox'))
        filenames.sort(key=lambda x: float(os.path.basename(x)[:-5]))

    plt.ion()
    for i, filename in enumerate(filenames):
        print('{}/{}'.format(i, len(filenames)))
        image_file = filename[:-5] + '.png'
        image = np.array(PIL.Image.open(image_file))
        bboxlist = vlslam_pb2.BoundingBoxList()
        with open(filename, 'rb') as fid:
            bboxlist.ParseFromString(fid.read())

        for bbox in bboxlist.bounding_boxes:
            pt1 = int(bbox.top_left_x), int(bbox.top_left_y)
            pt2 = int(bbox.bottom_right_x), int(bbox.bottom_right_y)

            if not bbox.HasField('azimuth'):
                azimuth_prob = estimate_azimuth(image[pt1[1]:pt2[1], pt1[0]:pt2[0], :], net)
                azimuth_mode = np.argmax(azimuth_prob)
                bbox.azimuth = azimuth_mode
                print('augment bbox with azimuth')
            else:
                azimuth_mode = bbox.azimuth

            if len(bbox.azimuth_prob) != 360:
                # clean up first
                while len(bbox.azimuth_prob) > 0:
                    del bbox.azimuth_prob[-1]
                azimuth_prob = estimate_azimuth(image[pt1[1]:pt2[1], pt1[0]:pt2[0], :], net)
                print "shape(azimuth_prob)=", azimuth_prob.shape
                for azi in azimuth_prob:
                    bbox.azimuth_prob.append(float(azi))
                print('augment bbox with azimuth probability')

            print "bbox azimuth prob size=", len(bbox.azimuth_prob)

            cv2.rectangle(image, pt1, pt2, (255, 205, 51), 2)
            cv2.putText(image, '%s: %.3f' % (bbox.class_name, bbox.scores[0]),
                        (pt1[0], pt1[1] + 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 0), thickness=2)
            cv2.putText(image, 'azimuth: %.3f' % (azimuth_mode),
                        (pt1[0], pt1[1] + 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 0), thickness=2)

        with open(filename, 'wb') as fid:
            fid.write(bboxlist.SerializeToString())

        plt.clf()
        plt.imshow(image)
        plt.pause(0.001)





