"""HW2 Basic Image Manipulation"""
from random import randint
from typing import Dict, List, Tuple
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


def plot_hist(hist: List[int], filename: str = "hist.jpg"):

    # fig = plt.figure(figsize=(10, 5))

    values = list(range(len(hist)))

    plt.bar(values, hist, width=0.1, color='blue')

    plt.xlabel("pixel value")
    plt.ylabel("count")
    plt.title("histogram")
    # plt.show()

    plt.savefig(filename)


def find_connected_components(img: np.ndarray, size_thres: int = 500, bin_val: int = 255) -> Dict[int, List[Tuple[int, int]]]:

    img = img.astype(np.float16)

    # fetches and initializes image dimensions
    h, w = img.shape[:2]

    discovered_flag = -1
    label = -2
    label_increment = -1

    components = {}

    queue = []
    for x in range(w):
        for y in range(h):

            if img[y, x] == bin_val:
                queue.append((x, y))

            while queue:
                x, y = queue.pop()

                img[y, x] = label

                if label not in components:
                    components[label] = []

                components[label].append((x, y))

                # checks 8 neighbors
                mvs = (-1, 0, 1)
                for x_mv in mvs:
                    for y_mv in mvs:
                        next_x = x + x_mv
                        next_y = y + y_mv

                        # excludes out of bound indices
                        if next_x < 0 or next_y < 0 or next_x >= w or next_y >= h:
                            continue

                        # excludes checked pixels
                        if img[next_y, next_x] != bin_val:
                            continue

                        img[next_y, next_x] = discovered_flag
                        queue.append((next_x, next_y))

            if label in components:
                label += label_increment

    components = {k: v for k, v in components.items() if len(v) >= size_thres}

    return components


def get_centroid(pixels: List[Tuple[int, int]]) -> Tuple[int, int]:

    sum_x = 0
    sum_y = 0
    for x, y in pixels:
        sum_x += x
        sum_y += y

    return (sum_x // len(pixels), sum_y // len(pixels))


def get_rect(pixels: List[Tuple[int, int]], height: int, width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:

    min_x = height
    min_y = width
    max_x = 0
    max_y = 0
    for x, y in pixels:
        max_x = max(x, max_x)
        max_y = max(y, max_y)
        min_x = min(x, min_x)
        min_y = min(y, min_y)

    return ((min_x, min_y), (max_x, max_y))


def draw_dot(img, pos, radius=2, color=(0, 0, 255)):
    cv2.circle(img, pos, radius=radius, color=color, thickness=-1)


def draw_cross(img, pos, radius: int = 5, color=(255, 255, 255), thickness: int = 2):
    cv2.line(img, (pos[0] - radius // 2, pos[1] - radius // 2), (pos[0] + radius // 2, pos[1] + radius // 2), color, thickness)
    cv2.line(img, (pos[0] - radius // 2, pos[1] + radius // 2), (pos[0] + radius // 2, pos[1] - radius // 2), color, thickness)


def draw_rect(img, start: Tuple[int, int], end: Tuple[int, int], color, thickness: int = 1):
    cv2.rectangle(img, start, end, color, thickness)


def draw_components(bin_img: np.ndarray, components: Dict[int, List[Tuple[int, int]]], max_val: int = 255):

    h, w = bin_img.shape[:2]

    result = np.dstack([bin_img, bin_img, bin_img])

    for _, v in components.items():
        color = (randint(max_val // 2, max_val), randint(max_val // 2, max_val), randint(max_val // 2, max_val))
        inv_color = tuple([max_val - val for val in color])

        for x, y in v:
            result[y, x] = color

        center = get_centroid(v)
        rect_start, rect_end = get_rect(v, h, w)

        draw_cross(result, center, 10, color=inv_color)
        draw_rect(result, rect_start, rect_end, color=inv_color, thickness=3)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    if op == "bin":
        img = binarize(img)
        cv2.imwrite("bin.jpg", img)

    elif op == "hist":
        hist = get_hist(img)
        plot_hist(hist)

    elif op == "comp":
        bin_img = binarize(img)
        components = find_connected_components(bin_img)
        result = draw_components(bin_img, components)
        cv2.imwrite("connected_components.jpg", result)

    else:
        raise Exception("unknown operation {}".format(op))

    # cv2.imshow("flipped", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
