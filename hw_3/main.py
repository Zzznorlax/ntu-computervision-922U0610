"""HW3 Histogram Equalization"""
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_hist(img: np.ndarray, size: int = 256) -> List[int]:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    hist = [0] * size

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                value = img[y, x, ch_idx]

                hist[value] += 1

    return hist


def plot_hist(hist: List[int], filename: str = "hist.jpg"):

    # fig = plt.figure(figsize=(10, 5))

    values = list(range(len(hist)))

    plt.bar(values, hist, width=0.1, color='blue')

    plt.xlabel("pixel value")
    plt.ylabel("count")
    plt.title("histogram")

    # plt.show()

    plt.savefig(filename)


def divide_intensity(img: np.ndarray, divider: float = 3) -> np.ndarray:
    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                img[y, x, ch_idx] = img[y, x, ch_idx] // divider

    return img


def equalize_hist(img: np.ndarray, max_intensity: int = 255) -> np.ndarray:
    hist = get_hist(img)

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    pixel_sum = sum(hist)

    accumulation = 0
    eq_map = {}

    for i in range(len(hist)):
        accumulation += hist[i] / pixel_sum
        eq_map[i] = accumulation * max_intensity

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                img[y, x, ch_idx] = eq_map[img[y, x, ch_idx]]

    return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    if op == "hist":
        hist = get_hist(img)
        plot_hist(hist)
        cv2.imwrite("original.jpg", img)

    elif op == "divide":
        divided_img = divide_intensity(img)
        hist = get_hist(divided_img)
        plot_hist(hist, filename="divided_hist.jpg")
        cv2.imwrite("divided.jpg", img)

    elif op == "hist_eq":
        divided_img = divide_intensity(img)
        equalized_img = equalize_hist(divided_img)
        hist = get_hist(equalized_img)
        plot_hist(hist, filename="equalized_hist.jpg")
        cv2.imwrite("equalized.jpg", equalized_img)

    else:
        raise Exception("unknown operation {}".format(op))

    # cv2.imshow("flipped", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
