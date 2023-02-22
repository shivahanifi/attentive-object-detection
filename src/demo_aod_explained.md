This is to clearly explain the modifications of the [online-vtd](https://github.com/shivahanifi/online-visual-target-detection/blob/main/src/demo_dev_online.py). 

## Table of Contents
- [Recap](#recap)
- [Goal](#goal)
- [Approach](#approach)
- [Errors](#errors)

                                   
## Recap
The previous explanations of this code can be used for understanding of the first part ([dev_online_code_explained.md](https://github.com/shivahanifi/online-visual-target-detection/blob/main/src/dev_online_code_explained.md),  [VT_Demo_Code.md](https://github.com/shivahanifi/visual-targets/blob/main/Demo/VT_Demo_Code.md)). Here only the modifications will be explained.

## Goal
The goal is to extract the bounding box of the concentrated area in the heatmap. This bounding box will be presented as output both visually and in an array form.

## Approach
- Thresholding

  The concept of thresholding will be used to extract the bounding box of the visually attended area. OpenCV has a specific function `cv.threshold` to apply the thresholding.

  The matter is straight-forward. For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.

  ```
  cv.threshold(src, thresh, maxval, type[, dst]) 
  ```
  - `src`: source image, which should be a grayscale image. 
  - `thresh`: threshold value which is used to classify the pixel values.
  - `maxval`: maximum value which is assigned to pixel values exceeding the threshold.
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

- Contour Features

  Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition. In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.
  ```
  cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
  ```
  - `image`: source image
  - `mode`: contour retrieval mode

    1. `cv.RETR_EXTERNAL`: retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours. 
    2. `cv.RETR_LIST`: 	retrieves all of the contours without establishing any hierarchical relationships. 
    3. `cv.RETR_CCOMP`: retrieves all of the contours and organizes them into a two-level hierarchy.
    4. `cv.RETR_TREE`: retrieves all of the contours and reconstructs a full hierarchy of nested contours. 
    5. ` cv.RETR_FLOODFILL`

  - `method` : contour approximation method.
    
    1.  `cv.CHAIN_APPROX_NONE`: stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
    2. `cv.CHAIN_APPROX_SIMPLE`: compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points. 
    3. `cv.CHAIN_APPROX_TC89_L1`
    4. `cv.CHAIN_APPROX_TC89_KCOS`


  It outputs the contours and hierarchy. Contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object. hierarchy contains information about the image topology. It has as many elements as the number of contours.

- Drawing Contours
  ```
  cv.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])
  ```
  It can also be used to draw any shape provided you have its boundary points.
  - `image`: source image
  - `contours`: the contours which should be passed as a Python list
  - `contourIdx`: index of contours (useful when drawing individual contour. To draw all contours, pass -1)
  - Remaining arguments are color, thickness etc

## Errors
1. YARP (namespace, detect, ports)

    For errors such as not detecting the YARP or ports or connection make sure to check the `yarp namespace` for both the docker and local machine to be the same. (After using it with iCub you need to change them back to the root). You may also use `yarp detect` to make sure the local machine can find YARP. A final check would be to check the visibility of ports by `yarp name list`.

2. `Unexpected error!!! src is not a numpy array, neither a scalar`
  
    Using the `raw_hm` which is the direct output of the model was causing this error since it was not converted to a Numpy array. using it after being converted to Numpy solved the problem.

3. `OpenCV Error: Unsupported format or combination of formats ([Start]FindContours supports only CV_8UC1 images when mode != CV_RETR_FLOODFILL otherwise supports CV_32SC1 images only)`

    This error message suggests that there is an issue with the format or combination of formats of the image being processed by the "FindContours" function in OpenCV. The shape of the raw_hm at this step is (1,1), which means (np.array([[a]]))