"""HW5 Mathematical Morphology - Gray Scaled Morphology"""
from typing import List
from unittest import result
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def dilation(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    kernel_h, kernel_w = kernel.shape[:2]

    result = np.zeros_like(img)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):

                max_val = 0
                for kernel_x in range(kernel_w):
                    for kernel_y in range(kernel_h):
                        if kernel[kernel_y, kernel_x] == 0:
                            continue

                        offset_x = kernel_x - kernel_w // 2
                        offset_y = kernel_y - kernel_h // 2

                        if y + offset_y >= h or y + offset_y < 0:
                            continue

                        if x + offset_x >= w or x + offset_x < 0:
                            continue

                        max_val = max(max_val, img[y + offset_y, x + offset_x, ch_idx])

                        result[y, x, ch_idx] = max_val

    return result


def erosion(img: np.ndarray, kernel: np.ndarray, bin_val: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    kernel_h, kernel_w = kernel.shape[:2]

    result = np.zeros_like(img)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):

                min_val = bin_val
                for kernel_x in range(kernel_w):
                    for kernel_y in range(kernel_h):
                        if kernel[kernel_y, kernel_x] == 0:
                            continue

                        offset_x = kernel_x - kernel_w // 2
                        offset_y = kernel_y - kernel_h // 2

                        if y + offset_y >= h or y + offset_y < 0:
                            continue

                        if x + offset_x >= w or x + offset_x < 0:
                            continue

                        min_val = min(min_val, img[y + offset_y, x + offset_x, ch_idx])

                        result[y, x, ch_idx] = min_val

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ])

    op = str(args.op)

    if op == "dilation":
        result = dilation(img, kernel=kernel)
        cv2.imwrite("dilation.jpg", result)

    elif op == "erosion":
        result = erosion(img, kernel=kernel)
        cv2.imwrite("erosion.jpg", result)

    elif op == "opening":
        result = erosion(img, kernel=kernel)
        result = dilation(result, kernel=kernel)
        cv2.imwrite("opening.jpg", result)

    elif op == "closing":
        result = dilation(img, kernel=kernel)
        result = erosion(result, kernel=kernel)
        cv2.imwrite("closing.jpg", result)

    else:
        raise Exception("unknown operation {}".format(op))

    # cv2.imshow("flipped", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
