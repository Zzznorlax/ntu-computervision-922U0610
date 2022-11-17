"""HW7 Thinning"""
from unittest import result
import cv2
import numpy as np
import argparse


def downsample(img: np.ndarray, scale: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    result = np.empty((h // scale, w // scale, ch), dtype=np.uint8)

    for ch_idx in range(ch):
        for x in range(w // scale):
            for y in range(h // scale):

                raw_x = x * scale
                raw_y = y * scale

                result[y, x, ch_idx] = img[raw_y, raw_x, ch_idx]

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


def yokoi(img: np.ndarray):
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    result = np.zeros_like(img)

    def h_op(b: int, c: int, d: int, e: int) -> str:
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'

        return 's'

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


def pair_relation(yokoi_img: np.ndarray, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    marked_img = np.zeros_like(img)

    def h_op(x_0: int, x_1: int, x_2: int, x_3: int, x_4: int, m: int = 1) -> int:
        if x_0 == m and (x_1 == m or x_2 == m or x_3 == m or x_4 == m):
            return 1
        return 2

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                x_0 = yokoi_img[y, x, ch_idx]

                x_1 = 0
                if x - 1 >= 0:
                    x_1 = yokoi_img[y, x - 1, ch_idx]

                x_2 = 0
                if x + 1 < w:
                    x_2 = yokoi_img[y, x + 1, ch_idx]

                x_3 = 0
                if y - 1 >= 0:
                    x_3 = yokoi_img[y - 1, x, ch_idx]

                x_4 = 0
                if y + 1 < h:
                    x_4 = yokoi_img[y + 1, x, ch_idx]

                marked_img[y, x, ch_idx] = h_op(x_0, x_1, x_2, x_3, x_4)

    return marked_img


def thinning(bin_img, img_pair):

    h, w = bin_img.shape[:2]

    def shrink_op(b: int, c: int, d: int, e: int) -> int:
        if b == c and (d != b or e != b):
            return 1

        return 0

    def kernel_op(bin_img: np.ndarray, x: int, y: int):
        if y == 0:
            if x == 0:
                x7, x2, x6 = 0, 0, 0
                x3, x0, x1 = 0, bin_img[y, x], bin_img[y, x + 1]
                x8, x4, x5 = 0, bin_img[y + 1, x], bin_img[y + 1, x + 1]

            elif x == w - 1:
                x7, x2, x6 = 0, 0, 0
                x3, x0, x1 = bin_img[y, x - 1], bin_img[y, x], 0
                x8, x4, x5 = bin_img[y + 1, x - 1], bin_img[y + 1, x], 0

            else:
                x7, x2, x6 = 0, 0, 0
                x3, x0, x1 = bin_img[y, x - 1], bin_img[y, x], bin_img[y, x + 1]
                x8, x4, x5 = bin_img[y + 1, x - 1], bin_img[y + 1, x], bin_img[y + 1, x + 1]

        elif y == h - 1:
            if x == 0:
                x7, x2, x6 = 0, bin_img[y - 1, x], bin_img[y - 1, x + 1]
                x3, x0, x1 = 0, bin_img[y, x], bin_img[y, x + 1]
                x8, x4, x5 = 0, 0, 0

            elif x == w - 1:
                x7, x2, x6 = bin_img[y - 1, x - 1], bin_img[y - 1, x], 0
                x3, x0, x1 = bin_img[y, x - 1], bin_img[y, x], 0
                x8, x4, x5 = 0, 0, 0

            else:
                x7, x2, x6 = bin_img[y - 1, x - 1], bin_img[y - 1, x], bin_img[y - 1, x + 1]
                x3, x0, x1 = bin_img[y, x - 1], bin_img[y, x], bin_img[y, x + 1]
                x8, x4, x5 = 0, 0, 0

        else:
            if x == 0:
                x7, x2, x6 = 0, bin_img[y - 1, x], bin_img[y - 1, x + 1]
                x3, x0, x1 = 0, bin_img[y, x], bin_img[y, x + 1]
                x8, x4, x5 = 0, bin_img[y + 1, x], bin_img[y + 1, x + 1]

            elif x == w - 1:
                x7, x2, x6 = bin_img[y - 1, x - 1], bin_img[y - 1, x], 0
                x3, x0, x1 = bin_img[y, x - 1], bin_img[y, x], 0
                x8, x4, x5 = bin_img[y + 1, x - 1], bin_img[y + 1, x], 0

            else:
                x7, x2, x6 = bin_img[y - 1, x - 1], bin_img[y - 1, x], bin_img[y - 1, x + 1]
                x3, x0, x1 = bin_img[y, x - 1], bin_img[y, x], bin_img[y, x + 1]
                x8, x4, x5 = bin_img[y + 1, x - 1], bin_img[y + 1, x], bin_img[y + 1, x + 1]

        a1 = shrink_op(x0, x1, x6, x2)
        a2 = shrink_op(x0, x2, x7, x3)
        a3 = shrink_op(x0, x3, x8, x4)
        a4 = shrink_op(x0, x4, x5, x1)

        if a1 + a2 + a3 + a4 == 1:
            return 0

        return x0

    bin_img = bin_img.copy()
    for y in range(h):
        for x in range(w):
            if bin_img[y, x] > 0 and img_pair[y, x] != 2:
                bin_img[y, x] = kernel_op(bin_img, x, y)
    return bin_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    if op == "thinning":
        img = downsample(img, scale=8)
        bin_img = binarize(img)

        thinned = bin_img
        while True:
            yokoi_mat = yokoi(thinned)
            marked = pair_relation(yokoi_mat, thinned)

            new_thinned = thinning(thinned, marked)

            if (thinned != new_thinned).sum() == 0:
                break

            thinned = new_thinned

        cv2.imwrite("thinning.jpg", thinned)

    else:
        raise Exception("unknown operation {}".format(op))
