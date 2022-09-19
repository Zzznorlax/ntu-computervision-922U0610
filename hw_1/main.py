"""HW1 Basic Image Manipulation"""
import cv2
import numpy as np
import argparse


def flip_vertically(img: np.ndarray) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h // 2):
                img[y, x, ch_idx], img[-y, x, ch_idx] = img[-y, x, ch_idx], img[y, x, ch_idx]

    return img


def flip_horizontally(img: np.ndarray) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    for ch_idx in range(ch):
        for x in range(w // 2):
            for y in range(h):
                img[y, x, ch_idx], img[y, -x, ch_idx] = img[y, -x, ch_idx], img[y, x, ch_idx]

    return img


def flip_diagonally(img: np.ndarray) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(x, h):
                img[y, x, ch_idx], img[x, y, ch_idx] = img[x, y, ch_idx], img[y, x, ch_idx]

    return img


def shrink(img: np.ndarray) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    idx_movement = (-1, 0, 1)

    result = np.empty((h // 2, w // 2, ch), dtype=np.uint8)

    for ch_idx in range(ch):
        for x in range(w // 2):
            for y in range(h // 2):

                kernel_idx_x = raw_x = x * 2
                kernel_idx_y = raw_y = y * 2
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


def rotate(img: np.ndarray, angle: float = 45):
    center = tuple(np.array(img.shape[1::-1]) / 2)

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    if op == "flip_v":
        img = flip_vertically(img)

    elif op == "flip_h":
        img = flip_horizontally(img)

    elif op == "flip_d":
        img = flip_diagonally(img)

    elif op == "shrink":
        img = shrink(img)

    elif op == "rot":
        img = rotate(img)

    elif op == "bin":
        img = binarize(img)

    else:
        raise Exception("unknown operation {}".format(op))

    cv2.imwrite("output.jpg", img)

    # cv2.imshow("flipped", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
