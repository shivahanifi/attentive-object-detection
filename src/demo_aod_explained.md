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
In this example, we start by loading an image and converting it to grayscale. We then apply a threshold to create a binary image where all pixels above the threshold are white and all pixels below the threshold are black.

Next, we use cv2.findContours() to find contours in the binary image. We loop over each contour and use cv2.contourArea() to check if the contour is large enough to be considered a valid region (you can adjust the threshold to your liking).

For each valid contour, we use cv2.boundingRect() to get the bounding box coordinates. We also set the contour pixels to white in a mask image using the cv2.drawContours() function.

Finally, we draw a green rectangle around each valid contour on the original image using cv2.rectangle(), but we only draw the rectangle where the corresponding pixels are white in the mask image. This ensures that the bounding boxes only appear around the white areas in the image.

We then display the image with bounding boxes using cv2.imshow() and wait for a key press before closing the window.