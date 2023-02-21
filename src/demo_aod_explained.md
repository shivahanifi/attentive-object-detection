This is to clearly explain the modifications of the [online-vtd](https://github.com/shivahanifi/online-visual-target-detection/blob/main/src/demo_dev_online.py). 

## Table of Contents
- [Recap](#recap)

                                   
## Recap
The previous explanations of this code can be used for understanding of the first part ([dev_online_code_explained.md](https://github.com/shivahanifi/online-visual-target-detection/blob/main/src/dev_online_code_explained.md),  [VT_Demo_Code.md](https://github.com/shivahanifi/visual-targets/blob/main/Demo/VT_Demo_Code.md)). Here only the modifications will be explained.

## Goal
The goal is to extract the bounding box of the concentrated area in the heatmap. This bounding box will be presented as output both visually and in an array form.

## Approach
The concept of thresholding will be used to extract the bounding box of the visually attended area. OpenCV has a specific function `cv.threshold` to apply the thresholding.

 The matter is straight-forward. For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.

```
cv.threshold(src, thresh, maxval, type[, dst]) 
```
- src: source image, which should be a grayscale image. 
- thresh: threshold value which is used to classify the pixel values.
- maxval: maximum value which is assigned to pixel values exceeding the threshold.
- OpenCV provides different types of thresholding which is given by the fourth parameter of the function. The thresholding types include:
  1. cv.THRESH_BINARY
  2. cv.THRESH_BINARY_INV
  3. cv.THRESH_TRUNC
  4. cv.THRESH_TOZERO
  5. cv.THRESH_TOZERO_INV

The method returns two outputs. The first is the threshold that was used and the second output is the thresholded image.

