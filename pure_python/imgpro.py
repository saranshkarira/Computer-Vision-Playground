#!/usr/bin/python3

import cv2  # Note: openCV is used to load Images ONLY!
import numpy as np  # NOTE: All the algorithms are written in pure loops, numpy is only used for list to image and vice versa conversions


import math
# import argparse
import sys
import random

# clamp the pixel values and convert all floats to ints
def clamp(x): return int(max(min(x, 255), 0))


# kernels
def gauss_kernel(k_size, sigma, pad):
    # center of kernel is the mean of gauss
    var = 2 * (sigma**2)
    kernel = [[0] * k_size for i in range(k_size)]
    norm = 0

    for i in range(-pad, k_size - pad):
        for j in range(-pad, k_size - pad):
            num = (i**2 + j**2)
            # Unnormalized Gaussian equation
            kernel[i + pad][j + pad] = (math.exp(-num / var))  # / math.sqrt(2 * (math.pi) * var)
            norm += kernel[i + pad][j + pad]

    # using sum to normalize instead of 1/pi*var
    for i in range(k_size):
        for j in range(k_size):
            kernel[i][j] /= norm
    return kernel

# Assuming kernels for sharp and edge
def sharp_kernel(k_size): return [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]


def edge_kernel(k_size): return [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]


def bilinear_kernel(k_size, pad):
    # Continous interpolation at each point x with value pad-x, in our case (pad-x)(pad-y)/2, 2 for normalization

    kernel = np.zeros([k_size, k_size, 3], dtype=np.int8).tolist()  # Numpy only for list initialization because  soft copy sucks!
    for i in range(-pad, k_size - pad):
        for j in range(-pad, k_size - pad):
            # print(j + pad)
            kernel[i + pad][j + pad] = ((pad - abs(i)) + (pad - abs(j))) // 2
    return kernel


# Noise
def noised(image, noise):
    noise = float(noise)
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] += [clamp(noise * random.randint(1, 255)) for i in range(3)]

    return image


# Luminance
# Brightness
def brighten(image, factor):
    factor = float(factor)
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = [clamp(pix_val * factor) for pix_val in image[i][j]]
    return image


# Contrast
def change_contrast(image, factor):
    factor = float(factor)
    # 255 to scale to pixel range
    # instead of using a median gray image, we know [128,128,128] will be gray, so using that instead
    factor = (259 * (factor * 255)) / (255 * (259 - factor))
    for i in range(len(image)):
        for j in range(len(image[0])):
            # image[i][j] = [clamp(128 + factor * (a - 128)) for a in image[i][j]] # Method 2
            # suggested interpolation method ((1-alpha)*gray)+(alpha*og_image)
            image[i][j] = [clamp((1 - factor) * 128 + factor * (a)) for a in image[i][j]]
    return image


# Linear Filtering
# Gaussian/ convolve the gaussian kernel to each pixel and the normal weighted sum of nearby pixel blurs
def blur(image, sigma):  # Implement Padding
    sigma = float(sigma)
    k_size = 5
    pad = k_size // 2
    kernel = gauss_kernel(k_size, sigma, pad)
    clone = image
    for i in range(pad, len(image) - pad):
        for j in range(pad, len(image[0]) - pad):
            conv = [0, 0, 0]
            for x in range(len(kernel)):
                for y in range(len(kernel[0])):
                    conv[0] += image[i - pad + x][j - pad + y][0] * kernel[x][y]
                    conv[1] += image[i - pad + x][j - pad + y][1] * kernel[x][y]
                    conv[2] += image[i - pad + x][j - pad + y][2] * kernel[x][y]
            clone[i][j] = [clamp(i) for i in conv]
    return clone


# Sharpen
# Similar to a mild edge detector, assuming alpha = 0.2
def sharpen(image):
    k_size = 3
    pad = k_size // 2
    kernel = sharp_kernel(k_size)
    clone = image
    for i in range(pad, len(image) - pad):
        for j in range(pad, len(image[0]) - pad):
            conv = [0, 0, 0]
            for x in range(len(kernel)):
                for y in range(len(kernel[0])):
                    conv[0] += image[i - pad + x][j - pad + y][0] * kernel[x][y]
                    conv[1] += image[i - pad + x][j - pad + y][1] * kernel[x][y]
                    conv[2] += image[i - pad + x][j - pad + y][2] * kernel[x][y]
            clone[i][j] = [clamp((conv[a] * 0.2) + (image[i][j][a])) for a in range(3)]
    return clone


# edge_detect
def edge_detect(image):
    k_size = 3
    pad = k_size // 2
    kernel = edge_kernel(k_size)
    clone = image
    for i in range(pad, len(image) - pad):
        for j in range(pad, len(image[0]) - pad):
            conv = [0, 0, 0]
            for x in range(len(kernel)):
                for y in range(len(kernel[0])):
                    conv[0] += image[i - pad + x][j - pad + y][0] * kernel[x][y]
                    conv[1] += image[i - pad + x][j - pad + y][1] * kernel[x][y]
                    conv[2] += image[i - pad + x][j - pad + y][2] * kernel[x][y]
            clone[i][j] = [clamp(i) for i in conv]
    return clone


# Resampling
# Scale
def scale(image, scale_x, scale_y, scale_type):
    # shapes of old and scale matrix
    scale_x = float(scale_x)
    scale_y = float(scale_y)
    og_shape = [len(image), len(image[0])]
    new_shape = [round(og_shape[0] * scale_x), round(og_shape[1] * scale_y)]

    # initialize output matrix with zeros
    out = np.zeros([*new_shape, 3]).tolist()

    # for gaussian map a point in output to a gaussian kernel in the input
    if scale_type == 'Gaussian':
        # kernel size = inverse of minification factor rounded to closest odd no greater than or equal to 3
        k_size = max(3, round(1 / scale_x) if round(1 / scale_x) % 2 == 1 else (round(1 / scale_x) + 1))
        pad = k_size // 2
        kernel = gauss_kernel(k_size, 0.5, pad)
        for i in range(new_shape[0] - 2 * pad):
            for j in range(new_shape[1] - 2 * pad):
                conv = [0, 0, 0]
                # Map of output pixels to input pixels
                ii = round((i - 1) * (og_shape[0] - 1) / (scale_x * og_shape[0] - 1) + 1)
                jj = round((j - 1) * (og_shape[1] - 1) / (scale_y * og_shape[1] - 1) + 1)
                for x in range(len(kernel)):
                    for y in range(len(kernel[0])):
                        # print(jj - pad + x, len(image[0]))
                        conv[0] += image[ii - pad + x][jj - pad + y][0] * kernel[x][y]
                        conv[1] += image[ii - pad + x][jj - pad + y][1] * kernel[x][y]
                        conv[2] += image[ii - pad + x][jj - pad + y][2] * kernel[x][y]  # Trying without padding since downsample
                out[i][j] = [clamp(i) for i in conv]

    elif scale_type == 'Point':

        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                conv = [0, 0, 0]
                # Mapping output image to input image
                # print((i - 1) * (og_shape[0] - 1) / (scale_x * og_shape[0] - 1) + 1)
                ii = round((i - 1) * (og_shape[0] - 1) / (scale_x * og_shape[0] - 1) + 1)
                jj = round((j - 1) * (og_shape[1] - 1) / (scale_y * og_shape[1] - 1) + 1)

                out[i][j] = image[ii][jj]

    # uses bilinear curve to convolve, the values maximize to center linearly
    elif scale_type == 'Bilinear':
        k_size = 3
        pad = k_size // 2
        kernel = bilinear_kernel(k_size, pad)
        for i in range(new_shape[0] - 2 * pad):
            for j in range(new_shape[1] - 2 * pad):
                conv = [0, 0, 0]

                ii = round((i - 1) * (og_shape[0] - 1) / (scale_x * og_shape[0] - 1) + 1)
                jj = round((j - 1) * (og_shape[1] - 1) / (scale_y * og_shape[1] - 1) + 1)
                for x in range(len(kernel)):
                    for y in range(len(kernel[0])):
                        conv[0] += image[ii - pad + x][jj - pad + y][0] * kernel[x][y]
                        conv[1] += image[ii - pad + x][jj - pad + y][1] * kernel[x][y]
                        conv[2] += image[ii - pad + x][jj - pad + y][2] * kernel[x][y]  # Trying without padding since downsample
                out[i][j] = [clamp(i) for i in conv]

    return out


# Composite
# Alpha interpolates between the two
def composite(image_base, image_top, mask, alpha):
    # We can also use bitwise operators instead
    # We'll initialize a zero output array replace zero with either of the image pixels using mask as conditional
    # out = [[[0] * 3] * len(image_base[0])] * len(image_base)
    alpha = float(alpha)
    image_top = cv2.imread(image_top).tolist()
    mask = cv2.imread(mask).tolist()
    out = image_base
    for i in range(len(out)):
        for j in range(len(out[0])):

            if mask[i][j][1] < 128:
                continue
            elif mask[i][j][1] > 128:
                out[i][j][0] = (alpha * out[i][j][0]) + ((1 - alpha) * image_top[i][j][0])
                out[i][j][1] = (alpha * out[i][j][1]) + ((1 - alpha) * image_top[i][j][1])
                out[i][j][2] = (alpha * out[i][j][2]) + ((1 - alpha) * image_top[i][j][2])
    return out


if __name__ == '__main__':
    # hard code arguments parsing because order matters as well as variable arguments
    checklist = sys.argv[1:]

    # classifying fxns based on no of args they take
    func_dict = {'-composite': composite, '-scale': scale, '-edge': edge_detect, '-blur': blur, '-sharpen': sharpen, '-contrast': change_contrast, '-brightness': brighten}
    zero_arg = ['-sharpen', '-edge']
    one_arg = ['-brightness', '-contrast', '-blur']
    three_arg = ['-scale', '-composite']

    # first two args are images
    image_in = checklist.pop(0)
    image_out = checklist.pop(0)
    image = cv2.imread(image_in).tolist()

    # passing the functions in the exact order arguments were passed
    for item in checklist:
        if item in func_dict:
            if item in zero_arg:
                image = func_dict[item](image)
            elif item in one_arg:
                print(checklist[checklist.index(item) + 1])
                image = func_dict[item](image, checklist[(checklist.index(item) + 1)])
            elif item in three_arg:
                image = func_dict[item](image, *checklist[checklist.index(item) + 1:checklist.index(item) + 4])

    image = np.asarray(image)
    cv2.imwrite(image_out, image)
