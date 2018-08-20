# OpenCV Playground:
Implementing multiple Computer Vision and Image Processing techniques using OpenCV

## Requirements:
- Google Colab: For Notebooks(Not necessary, just don't run the setup part)
- Python3.6
- OpenCV3
- Numpy

## Implementations:

### Pure Python: No OpenCV, No Numpy:
- [x] Edge Detection
- [x] Image Sharpening
- [x] Brightness
- [x] Contrast
- [x] Blur
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

### Workaround : Image Processing Techniques in Pure Python[No numpy]

## Requirements: OpenCV3 : To load the images only

## Note: No library is used to implement either part or whole of the algorithms. Not even numpy.

- The directory for this implementation is `pure_python/`, `cd` to this dir.
- The file `imgpro.py` contains the implementations, while `make.py` generates some output samples with contrasting params.
- To use the code in any new terminal session, either run `. ./setup.sh` or `source setup.sh`
- Now we have two commands available. `generate` and `imgpro`
- `generate` needs no flags, run it and all the output images for the writeup will be generated.

- The first two arguments to `imgpro` are paths to input image and output image respectively.
- The 3rd argument is the type of processing we want to use and then it's parameters
- We can implement multiple different processing techniques on a single image in the sequenced order of writing arguments.

    - example `imgpro input/princeton_small.jpg  abc.jpg -edge -blur 1.5`

##funtions and args:

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

### An Example of Canny Edge Detection: 
<img src="https://raw.githubusercontent.com/saranshkarira/opencv-playground/master/src/canny_edge_detector_sample.png">

**Adios!**
