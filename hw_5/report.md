# Computer Vision Homework 5

## Mathematical Morphology - Gray Scaled Morphology

**R11525079 游子霆**

### Description
In this homework, a program that can be used to perform
(a) Dilation
(b) Erosion
(c) Opening
(d) Closing

### Part 1.

**a. Dilation**
By going through each pixel and use the maximum value in the pixel set of kernel as the new value
```python
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
```
A dilated image can be generated using the following command:
```shell
python3 hw_5/main.py --img=inputs/lena.bmp --op=dilation
```
![dilation.jpg](assets/dilation.jpg)

**b. Erosion**
By going through each pixel and use the minimum value in the pixel set of kernel as the new value
```python
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
```
A eroded image can be generated using the following command:
```shell
python3 hw_5/main.py --img=inputs/lena.bmp --op=erosion
```
![erosion.jpg](assets/erosion.jpg)

**c. Opening**
By performing erosion and then dilation we can generate the result
```python
        result = erosion(img, kernel=kernel)
        result = dilation(result, kernel=kernel)
```
The image after opening operation is performed can be generated using the following command:
```shell
python3 hw_5/main.py --img=inputs/lena.bmp --op=opening
```
![opening.jpg](assets/opening.jpg)

**d. Closing**
By performing dilation and then erosion we can generate the result
```python
        result = dilation(result, kernel=kernel)
        result = erosion(img, kernel=kernel)
```
The image after closing operation is performed can be generated using the following command:
```shell
python3 hw_5/main.py --img=inputs/lena.bmp --op=closing
```
![closing.jpg](assets/closing.jpg)
