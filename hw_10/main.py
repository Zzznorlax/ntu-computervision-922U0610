import cv2
import numpy as np
import argparse


def conv(img: np.ndarray, kernel: np.ndarray) -> float:

    img_h, img_w = img.shape[:2]
    k_h, k_w = kernel.shape[:2]

    res = 0
    for i in range(img_h):
        for j in range(img_w):
            if img_h - i - 1 >= 0 and img_h - i - 1 < k_h and img_w - j - 1 >= 0 and img_w - j - 1 < k_w:
                res += img[i, j] * kernel[img_h - i - 1, img_w - j - 1]

    return res


def zero_crossing(img: np.ndarray, kernel: np.ndarray, thres: float = 1, bin_val: int = 255) -> np.ndarray:

    mask = np.zeros_like(img)
    res = np.ones_like(img)

    h, w = img.shape[:2]

    k_h, k_w = kernel.shape[:2]

    mask_neg_val = 2

    for y in range(h):
        for x in range(w):

            val = conv(img[y:y + k_h, x:x + k_w], kernel)
            output = 0

            if val >= thres:
                output = 1

            elif val <= -thres:
                output = mask_neg_val

            mask[y, x] = output

    idx_mvs = [-1, 0, 1]
    for y in range(h):
        for x in range(w):

            if mask[y, x] != 1:
                continue

            for mv_y in idx_mvs:
                if res[y, x] == 0:
                    break

                for mv_x in idx_mvs:
                    if mv_x + x < 0 or mv_x + x >= w or mv_y + y < 0 or mv_y + y >= h:
                        continue

                    if mask[y + mv_y, x + mv_x] == mask_neg_val:
                        res[y, x] = 0
                        break

    return res * bin_val


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")
    parser.add_argument("--thres")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    lap_1_k = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    lap_2_k = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])

    min_var_lap_k = np.array([
        [2, -1, 2],
        [-1, -4, -1],
        [2, -1, 2],
    ])

    lap_of_g_k = np.array([
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0, ],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0, ],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0, ],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1, ],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1, ],
        [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2, ],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1, ],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1, ],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0, ],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0, ],
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0, ],
    ])

    diff_of_g_k = np.array([
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1, ],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3, ],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4, ],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6, ],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7, ],
        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8, ],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7, ],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6, ],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4, ],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3, ],
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1, ]
    ])

    if op == "lap_1":
        thres = 30
        img = zero_crossing(img, kernel=lap_1_k, thres=thres)

        cv2.imwrite("{}.jpg".format(op), img)

    elif op == "lap_2":
        thres = 50
        img = zero_crossing(img, kernel=lap_2_k, thres=thres)

        cv2.imwrite("{}.jpg".format(op), img)

    elif op == "mv":
        thres = 55
        img = zero_crossing(img, kernel=min_var_lap_k, thres=thres)

        cv2.imwrite("{}.jpg".format(op), img)

    elif op == "log":
        thres = 3000
        img = zero_crossing(img, kernel=lap_of_g_k, thres=thres)

        cv2.imwrite("{}.jpg".format(op), img)

    elif op == "dog":
        thres = 1
        img = zero_crossing(img, kernel=diff_of_g_k, thres=thres)

        cv2.imwrite("{}.jpg".format(op), img)

    else:
        raise Exception("unknown operation {}".format(op))
