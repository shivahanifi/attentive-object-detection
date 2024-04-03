import os
import struct
import glob
import argparse

import cv2
import numpy as np


def read_depth(filename):
    with open(filename, 'rb') as f:
        width = f.read(8)
        width = int.from_bytes(width, "little")
        assert width == 640
        height = f.read(8)
        height = int.from_bytes(height, "little")
        assert height == 480

        depth_img = []
        while (True):
            depthval_b = f.read(4)      # binary, little endian
            if not depthval_b:
                break
            depthval_m = struct.unpack("<f", depthval_b)    # depth val as meters
            depth_img.append(depthval_m)
        assert len(depth_img) == height * width

    depth_img = np.array(depth_img).reshape(height, width)

    return depth_img


def main(path_to_float_dir):
    files = glob.glob(os.path.join(path_to_float_dir, '*.float'))
    files.sort()

    for filename in files:
        depth_img = read_depth(filename)

        # normalize and show
        temp = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        cv2.imshow("depth", temp)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_float_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.path_to_float_dir)
