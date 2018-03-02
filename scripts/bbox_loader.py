from vlslam_tools import vlslam_pb2
import argparse
import glob
import os
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import cv2
from vlslam_tools.utils import load_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/mnt/external/tmp/swivel_chair',
                        help='dataset root')
    args = parser.parse_args()

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

        bboxlist = load_bbox(filename)

        for bbox in bboxlist.bounding_boxes:
            pt1 = int(bbox.top_left_x), int(bbox.top_left_y)
            pt2 = int(bbox.bottom_right_x), int(bbox.bottom_right_y)
            cv2.rectangle(image, pt1, pt2, (255, 205, 51), 2)
            cv2.putText(image, '%s: %.3f' % (bbox.class_name, bbox.scores[0]),
                        (pt1[0], pt1[1] + 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 0), thickness=2)
            if bbox.HasField('azimuth'):
                cv2.putText(image, 'azimuth: %.3f' % float(bbox.azimuth),
                            (pt1[0], pt1[1] + 30),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0, (255, 255, 0), thickness=2)

        plt.clf()
        plt.imshow(image)
        plt.pause(0.01)






