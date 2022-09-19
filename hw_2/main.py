"""HW2 Basic Image Manipulation"""
from turtle import color
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def binarize(img: np.ndarray, thres: int = 128, upper_val: int = 255, lower_val: int = 0) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                value = img[y, x, ch_idx]

                if value >= thres:
                    img[y, x, ch_idx] = upper_val
                else:
                    img[y, x, ch_idx] = lower_val

    return img


def get_hist(img: np.ndarray, size: int = 256) -> List[int]:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    hist = [0] * size

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):
                value = img[y, x, ch_idx]

                hist[value] += 1

    return hist


def plot_hist(hist: List[int]):

    # fig = plt.figure(figsize=(10, 5))

    values = list(range(len(hist)))

    plt.bar(values, hist, width=0.1, color='blue')

    plt.xlabel("pixel value")
    plt.ylabel("count")
    plt.title("histogram")
    plt.show()



def dfs(graph: np.ndarray, x: int, y: int, group: int, count: int) -> int:

    if graph[y, x] != 1:
        return count

    graph[y, x] = group
    count += 1

    h, w = graph.shape[:-2]

    mvs = (-1, 0, 1)
    for x_mv in mvs:
        for y_mv in mvs:
            next_x = x + x_mv
            next_y = y + y_mv

            if next_x < 0 or next_y < 0 or next_x >= w or next_y >= h:
                continue

            count = dfs(graph, next_x, next_y, group, count)

    return count


def find_connected_components(graph: np.ndarray, size_thres: int = 500) -> np.ndarray:



if __name__ == '__main__':

    img_path = "inputs/lena.bmp"
    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    hist = get_hist(img)
    plot_hist(hist)
