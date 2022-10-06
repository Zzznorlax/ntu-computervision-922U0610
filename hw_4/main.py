"""HW3 Histogram Equalization"""
from typing import List
from unittest import result
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


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


def invert(img: np.ndarray, max_val: int = 255) -> np.ndarray:
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
                result[y, x, ch_idx] = max_val - img[y, x, ch_idx]

    return result


def dilation(img: np.ndarray, kernel: np.ndarray, binary_val: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    kernel_h, kernel_w = kernel.shape[:2]

    result = np.array(img, copy=True)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):

                if img[y, x, ch_idx] == binary_val:

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

                            result[y + offset_y, x + offset_x] = binary_val

    return result


def erosion(img: np.ndarray, kernel: np.ndarray, binary_val: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    kernel_h, kernel_w = kernel.shape[:2]

    result = np.array(img, copy=True)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):

                result[y, x, ch_idx] = binary_val

                for kernel_x in range(kernel_w):
                    for kernel_y in range(kernel_h):

                        offset_x = kernel_x - kernel_w // 2
                        offset_y = kernel_y - kernel_h // 2

                        if kernel[kernel_y, kernel_x] == 0:
                            continue

                        if y + offset_y >= h or y + offset_y < 0:
                            continue

                        if x + offset_x >= w or x + offset_x < 0:
                            continue

                        # checks for foreground structure
                        if kernel[kernel_y, kernel_x] == 1 and img[y + offset_y, x + offset_x, ch_idx] == 0:
                            result[y, x, ch_idx] = 0
                            break

                        # checks for background structure
                        if kernel[kernel_y, kernel_x] == -1 and img[y + offset_y, x + offset_x, ch_idx] == binary_val:
                            result[y, x, ch_idx] = 0
                            break

                    if result[y, x, ch_idx] == 0:
                        break

    return result


def intersect(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    result = np.zeros_like(img_a)

    h, w = result.shape[:2]
    ch = 1
    if len(result.shape) > 2:
        ch = result.shape[2]
    elif len(result.shape) == 2:
        result = np.expand_dims(result, axis=-1)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                if img_a[y, x, ch_idx] == img_b[y, x, ch_idx] and img_a[y, x, ch_idx] > 0:
                    result[y, x, ch_idx] = img_a[y, x, ch_idx]

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
    img = binarize(img)
    cv2.imwrite("bin.jpg", img)

    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ])

    kernel_j = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    kernel_k = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
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

    elif op == "hit_n_miss":

        result_j = erosion(img, kernel=kernel_j)

        inverted = invert(img)
        result_k = erosion(inverted, kernel=kernel_k)

        result = intersect(result_j, result_k)

        # cv2.imwrite("inv.jpg", inverted)
        # cv2.imwrite("j.jpg", result_j)
        # cv2.imwrite("k.jpg", result_k)

        cv2.imwrite("hit_or_miss.jpg", result)

    else:
        raise Exception("unknown operation {}".format(op))

    # cv2.imshow("flipped", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
