from imgpro import *
import cv2
import numpy as np


def write(image, name):
    print('writing ' + name)
    name = 'output/' + name
    cv2.imwrite(name, np.asarray(image))


def read(name):
    name = 'input/' + name
    return cv2.imread(name).tolist()


def fast_fxn(in_image, out_image, fxn, params):
    image = fxn(in_image, *params)
    write(image, out_image)


# zero args
# Sharpen
image = read('princeton_small.jpg')
image = sharpen(image)
write(image, 'sharpen.jpg')

# Edge Detect
image = read('princeton_small.jpg')
image = edge_detect(image)
write(image, 'edgedetect.jpg')


# # one args

# Blur
image = read('princeton_small.jpg')
fast_fxn(image, 'blur_0.125.jpg', blur, [0.125])
fast_fxn(image, 'blur_2.jpg', blur, [2])
fast_fxn(image, 'blur_8.jpg', blur, [8])

# Brightness
fast_fxn(image, 'princeton_small_brightness_0.0.jpg', brighten, [0.0])
image = read('princeton_small.jpg')
fast_fxn(image, 'princeton_small_brightness_0.5.jpg', brighten, [0.5])
image = read('princeton_small.jpg')  # rereading images because of softcopy
fast_fxn(image, 'princeton_small_brightness_2.0.jpg', brighten, [2.0])

# Contrast
image = read('c.jpg')
fast_fxn(image, 'c_contrast_2.0.jpg', change_contrast, [2.0])
fast_fxn(image, 'c_contrast_0.5.jpg', change_contrast, [0.5])
fast_fxn(image, 'c_contrast_-0.5.jpg', change_contrast, [-0.5])

fast_fxn(image, 'c_contrast_0.0.jpg', change_contrast, [0.0])


# # three args
# Scale
image = read('scaleinput.jpg')
fast_fxn(image, 'scale_gaussian.jpg', scale, [0.3, 0.3, 'Gaussian'])
image = read('scaleinput.jpg')
fast_fxn(image, 'scale_point.jpg', scale, [0.3, 0.3, 'Point'])
image = read('scaleinput.jpg')
fast_fxn(image, 'scale_bilinear.jpg', scale, [0.3, 0.3, 'Bilinear'])

# Composite
image = read('comp_background.jpg')
params = ['input/comp_foreground.jpg', 'input/comp_mask.jpg', 0]
fast_fxn(image, 'composite.jpg', composite, params)
