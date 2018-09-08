# OpenCV Playground:
Implementing multiple Computer Vision and Image Processing techniques using OpenCV

### Some Examples:
**TODO** : Generate outputs of Implemenatations of Script Files as Well

#### Binary Masks and Filtering
<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/bitwise_og.png" width="250" height="250" />

<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/binary_thresh_mask.png" width="150" height="200" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/bitwise_mask_filtering.png" width="150" height="200" />

#### Foreground Extraction
<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/fg_extraction_grabcut.png" width="150" height="200" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/fg_extraction2.png" width="150" height="200" />

#### BruteForce Feature Matching
<img src="https://github.com/saranshkarira/opencv-playground/blob/master/src/bruteforce_feature_matching.png" />

#### Canny Edge Detection: 
<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/canny_edge_detector_sample.png" width="300" height="256" />

#### Corner Detection
<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/corner_detection.png" width="256" height="256" />

#### Thresholding
 - **ORIGNAL** (Not so Clear) | **GRAY**
 
 <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_og.png" width="400" height="225" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_togray.png" width="400" height="225" />
 
 - **BINARY** | **BINARY + GRAY**
 
 <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_binary.png" width="400" height="225" /> <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_binary_after_gray.png" width="400" height="225" />
 
 - **TRUNCATED** | **TO ZERO** 
 
 <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_trunc.png" width="400" height="225" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_tozero.png" width="400" height="225" />
 
 - **ADAPTIVE : GAUSSIAN** | **OTSU**
 
 <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_adaptive_gaus.png" width="400" height="225" /> <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/thresh_adaptive_otsu.png" width="400" height="225" />
 
### Examples of Pure Python Implementations:
 - **Gaussian Blur 0.125| 2 | 8**
 
 <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/blur_0.125.jpg" width="150" height="128" /> <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/blur_2.jpg" width="150" height="128" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/blur_8.jpg" width="150" height="128" />
 
- **Edge Detection**

<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/edgedetect.jpg" width="150" height="128" />

- **Sharpen**

<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/input/princeton_small.jpg" width="150" height="128" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/sharpen.jpg" width="150" height="128" />

- **Contrast -0.5 | 0 | 0.5 |2**

<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/input/c.jpg" width="300" height="256" />

<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/c_contrast_-0.5.jpg" width="150" height="128" /> <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/c_contrast_0.0.jpg" width="150" height="128" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/c_contrast_0.5.jpg" width="150" height="128" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/c_contrast_2.0.jpg" width="150" height="128" />

- **Composite**

- **Brightness 0.0|0.5|2.0**

 <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/princeton_small_brightness_0.0.jpg" width="150" height="128" /> <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/princeton_small_brightness_0.5.jpg" width="150" height="128" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/princeton_small_brightness_2.0.jpg" width="150" height="128" />

- **Scale BiLinear | Gaussian | Point [Interpolation]**

<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/scale_bilinear.jpg" /> <img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/scale_gaussian.jpg" /><img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/pure_python/output/scale_point.jpg" />


## Implementations:

### Pure Python: No OpenCV, No Numpy:
- [x] Edge Detection
- [x] Image Sharpening
- [x] Brightness
- [x] Contrast
- [x] Gaussian Blur
- [x] Gaussian Interpolation
- [x] Bilinear Interpolation
- [x] Point Interpolation
- [x] Composite

### OpenCV:
- [x] Overlays
- [x] Thresholding
- [x] Creating Masks
- [x] Color Filtering
- [x] Smoothing
- [x] Morphological Transformations
- [x] Gradients
- [x] Edge Detections
- [x] Template Matching
- [x] GrabCut
- [x] Foreground/Motion Extraction
- [x] Corner Detection
- [x] Feature Matching by BruteForce(HomoGraphy)
- [x] Background Reduction
- [x] Haar Cascades
- [ ] Manual Haar Cascades


## Requirements:
- Google Colab: For Notebooks(Not necessary, just don't run the setup part)
- Python3.6
- OpenCV3
- Numpy


**Note**: No library is used to implement either part or whole of the algorithms. Not even numpy.

- The directory for this implementation is `pure_python/`, `cd` to this dir.
- The file `imgpro.py` contains the implementations, while `make.py` generates some output samples with contrasting params.
- To use the code in any new terminal session, either run `. ./setup.sh` or `source setup.sh`
- Now we have two commands available. `generate` and `imgpro`
- `generate` needs no flags, run it and all the output images for the writeup will be generated.

- The first two arguments to `imgpro` are paths to input image and output image respectively.
- The 3rd argument is the type of processing we want to use and then it's parameters
- We can implement multiple different processing techniques on a single image in the sequenced order of writing arguments.

    - Example `imgpro input/princeton_small.jpg  abc.jpg -edge -blur 1.5`

## funtions and args:

### Zero Arg Functions:
- `-edge`
- `-sharpen`

### Single Arg Functions:
- `-brighteness factor`
- `-contrast factor`
- `-blur factor`

### Three Args Functions:
- `-scale x_scale, y_scale, interpolation_type`
- `-composite fg_image, mask_image, alpha`
*Note* : The bg_image of Composite is the input image


### Workaround : OpenCV
- Some techniques are written in Colab and some are written as scripts, decision based on technique's computation needs as well as whether input was video/cam or an Image
- Colab Notebooks are stored in Notebooks/ ; Scripts are stored in Scripts/
- Processed and Raw Images are prefetched in Notebooks, while for scripts you'll need to supply your own feed, most likely video or a webcam feed.



**Adios!**
