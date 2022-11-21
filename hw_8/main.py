"""HW8 Noise Removal"""
import cv2
import numpy as np
import argparse
import math


def gaussian_noise(img: np.ndarray, mu, sigma, amp) -> np.ndarray:
    return img + amp * np.random.normal(mu, sigma, img.shape)


def salt_pepper_noise(img: np.ndarray, prob: float) -> np.ndarray:
    distribution_map: np.ndarray = np.random.uniform(0, 1, img.shape)  # type: ignore
    res = np.copy(img)
    h, w = img.shape

    for i in range(h):
        for j in range(w):
            if distribution_map[i, j] < prob:
                res[i, j] = 0
            elif distribution_map[i, j] > 1 - prob:
                res[i, j] = 255

    return res


def box_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    result = np.zeros_like(img)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):

                k_sum = 0
                for k_offset_x in range(kernel_size):
                    for k_offset_y in range(kernel_size):

                        offset_x = x + (k_offset_x - kernel_size // 2)
                        offset_y = y + (k_offset_y - kernel_size // 2)

                        if offset_x < 0 or offset_x >= w:
                            continue

                        if offset_y < 0 or offset_y >= h:
                            continue

                        k_sum += img[offset_y, offset_x, ch_idx]

                result[y, x, ch_idx] = k_sum // kernel_size**2

    return result


def median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:

    h, w = img.shape[:2]
    ch = 1
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        img = np.expand_dims(img, axis=-1)

    result = np.zeros_like(img)

    for ch_idx in range(ch):
        for x in range(w):
            for y in range(h):

                val_list = []
                for k_offset_x in range(kernel_size):
                    for k_offset_y in range(kernel_size):

                        offset_x = x + (k_offset_x - kernel_size // 2)
                        offset_y = y + (k_offset_y - kernel_size // 2)

                        if offset_x < 0 or offset_x >= w:
                            continue

                        if offset_y < 0 or offset_y >= h:
                            continue

                        val_list.append(img[offset_y, offset_x, ch_idx])

                val_list.sort()
                result[y, x, ch_idx] = val_list[len(val_list) // 2]

    return result


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


def opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = erosion(img, kernel)
    result = dilation(img, kernel)
    return result


def closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = dilation(img, kernel)
    result = erosion(img, kernel)
    return result


def snr(gt_img: np.ndarray, noise: np.ndarray) -> float:

    h, w = gt_img.shape[:2]

    gt_img = gt_img / 255
    noise = noise / 255

    mu_gt_img = 0
    power_gt_img = 0
    mu_noise = 0
    power_noise = 0

    for j in range(w):
        for i in range(h):
            mu_gt_img = mu_gt_img + gt_img[i, j]
            mu_noise = mu_noise + (noise[i, j] - gt_img[i, j])
    mu_gt_img = mu_gt_img / (h * w)
    mu_noise = mu_noise / (h * w)

    for i in range(w):
        for j in range(h):
            power_gt_img = power_gt_img + math.pow(gt_img[i, j] - mu_gt_img, 2)
            power_noise = power_noise + math.pow(noise[i, j] - gt_img[i, j] - mu_noise, 2)
    power_gt_img = power_gt_img / (h * w)
    power_noise = power_noise / (h * w)

    res = 20 * math.log10(math.sqrt(power_gt_img) / math.sqrt(power_noise))

    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic image manipulation program')
    parser.add_argument("--img")
    parser.add_argument("--op")
    parser.add_argument("--sample")

    args = parser.parse_args()

    img_path = args.img

    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

    op = str(args.op)

    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ])

    if op == "gen":

        noise_dict = {}
        noise_dict['a_10'] = gaussian_noise(img, 0, 1, 10)
        noise_dict['a_30'] = gaussian_noise(img, 0, 1, 30)

        noise_dict['b_10'] = salt_pepper_noise(img, 0.1)
        noise_dict['b_5'] = salt_pepper_noise(img, 0.05)

        box_filtered = {}
        median_filtered = {}
        o_c_filtered = {}
        c_o_filtered = {}
        for k, noise in noise_dict.items():
            box_filtered["c_3_{}".format(k)] = box_filter(noise, 3)
            box_filtered["c_5_{}".format(k)] = box_filter(noise, 5)

            median_filtered["d_3_{}".format(k)] = median_filter(noise, 3)
            median_filtered["d_5_{}".format(k)] = median_filter(noise, 5)

            o_c_filtered["e_oc_{}".format(k)] = opening(closing(noise, kernel), kernel)
            c_o_filtered["e_co_{}".format(k)] = closing(opening(noise, kernel), kernel)

        for k, noise in (noise_dict | box_filtered | median_filtered | o_c_filtered | c_o_filtered).items():
            print("{} SNR: {}".format(k, snr(img, noise)))
            cv2.imwrite("{}.jpg".format(k), noise)

    elif op == "snr":
        sample = str(args.sample)
        sample_img = cv2.imread(sample, flags=cv2.IMREAD_UNCHANGED)

        print("SNR: {}".format(snr(img, sample_img)))

    else:
        raise Exception("unknown operation {}".format(op))
