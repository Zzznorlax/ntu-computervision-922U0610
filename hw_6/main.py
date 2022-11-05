"""HW5 Mathematical Morphology - Gray Scaled Morphology"""
from typing import List
from unittest import result
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def downsample(img: np.ndarray, scale: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    idx_movement = (-1, 0, 1)

    result = np.zeros_like(img)

    result = np.empty((h // scale, w // scale, ch), dtype=np.uint8)

    for ch_idx in range(ch):
        for x in range(w // scale):
            for y in range(h // scale):

                kernel_idx_x = raw_x = x * scale
                kernel_idx_y = raw_y = y * scale
                kernel_sum = 0
                available_count = 0
                for x_mv in idx_movement:
                    kernel_idx_x = raw_x + x_mv
                    if kernel_idx_x < 0 or kernel_idx_x >= w:
                        continue

                    for y_mv in idx_movement:
                        kernel_idx_y = raw_y + y_mv
                        if kernel_idx_y < 0 or kernel_idx_y >= h:
                            continue

                        available_count += 1
                        kernel_sum += img[kernel_idx_y, kernel_idx_x, ch_idx]

                # uses the average of neighbor pixels as result value
                result[y, x, ch_idx] = kernel_sum / available_count

    return result


def binarize(img: np.ndarray, thres: int = 128, upper_val: int = 255, lower_val: int = 0) -> np.ndarray:

    h, w = img.shape[:2]

    for x in range(w):
        for y in range(h):
            value = img[y, x]

            if value >= thres:
                img[y, x] = upper_val
            else:
                img[y, x] = lower_val

    return img


def h_op(b: int, c: int, d: int, e: int) -> str:
    if b == c and (d != b or e != b):
        return 'q'
    if b == c and (d == b and e == b):
        return 'r'

    return 's'


def yokoi(img: np.ndarray):
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    result = np.zeros_like(img)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                if img[y, x, ch_idx] > 0:
                    if y == 0:
                        if x == 0:
                            x7, x2, x6 = 0, 0, 0
                            x3, x0, x1 = 0, img[y, x, ch_idx], img[y, x + 1, ch_idx]
                            x8, x4, x5 = 0, img[y + 1, x, ch_idx], img[y + 1, x + 1, ch_idx]

                        elif x == w - 1:
                            x7, x2, x6 = 0, 0, 0
                            x3, x0, x1 = img[y, x - 1, ch_idx], img[y, x, ch_idx], 0
                            x8, x4, x5 = img[y + 1, x - 1, ch_idx], img[y + 1, x, ch_idx], 0

                        else:
                            x7, x2, x6 = 0, 0, 0
                            x3, x0, x1 = img[y, x - 1, ch_idx], img[y, x, ch_idx], img[y, x + 1, ch_idx]
                            x8, x4, x5 = img[y + 1, x - 1, ch_idx], img[y + 1, x, ch_idx], img[y + 1, x + 1, ch_idx]

                    elif y == h - 1:
                        if x == 0:
                            x7, x2, x6 = 0, img[y - 1, x, ch_idx], img[y - 1, x + 1, ch_idx]
                            x3, x0, x1 = 0, img[y, x, ch_idx], img[y, x + 1, ch_idx]
                            x8, x4, x5 = 0, 0, 0

                        elif x == w - 1:
                            x7, x2, x6 = img[y - 1, x - 1, ch_idx], img[y - 1, x, ch_idx], 0
                            x3, x0, x1 = img[y, x - 1, ch_idx], img[y, x, ch_idx], 0
                            x8, x4, x5 = 0, 0, 0

                        else:
                            x7, x2, x6 = img[y - 1, x - 1, ch_idx], img[y - 1, x, ch_idx], img[y - 1, x + 1, ch_idx]
                            x3, x0, x1 = img[y, x - 1, ch_idx], img[y, x, ch_idx], img[y, x + 1, ch_idx]
                            x8, x4, x5 = 0, 0, 0
                    else:
                        if x == 0:
                            x7, x2, x6 = 0, img[y - 1, x, ch_idx], img[y - 1, x + 1, ch_idx]
                            x3, x0, x1 = 0, img[y, x, ch_idx], img[y, x + 1, ch_idx]
                            x8, x4, x5 = 0, img[y + 1, x, ch_idx], img[y + 1, x + 1, ch_idx]

                        elif x == w - 1:
                            x7, x2, x6 = img[y - 1, x - 1, ch_idx], img[y - 1, x, ch_idx], 0
                            x3, x0, x1 = img[y, x - 1, ch_idx], img[y, x, ch_idx], 0
                            x8, x4, x5 = img[y + 1, x - 1, ch_idx], img[y + 1, x, ch_idx], 0

                        else:
                            x7, x2, x6 = img[y - 1, x - 1, ch_idx], img[y - 1, x, ch_idx], img[y - 1, x + 1, ch_idx]
                            x3, x0, x1 = img[y, x - 1, ch_idx], img[y, x, ch_idx], img[y, x + 1, ch_idx]
                            x8, x4, x5 = img[y + 1, x - 1, ch_idx], img[y + 1, x, ch_idx], img[y + 1, x + 1, ch_idx]

                    a1 = h_op(x0, x1, x6, x2)
                    a2 = h_op(x0, x2, x7, x3)
                    a3 = h_op(x0, x3, x8, x4)
                    a4 = h_op(x0, x4, x5, x1)

                    if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                        result[y, x, ch_idx] = 5

                    else:
                        num = 0
                        for a_i in [a1, a2, a3, a4]:
                            if a_i == 'q':
                                num += 1

                        result[y, x, ch_idx] = num

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    if op == "yokoi":
        result = downsample(img, scale=8)
        result = binarize(result)
        yokoi_mat = np.squeeze(yokoi(result))

        np.savetxt("yokoi_mat.txt", yokoi_mat, fmt='%i')

    else:
        raise Exception("unknown operation {}".format(op))
