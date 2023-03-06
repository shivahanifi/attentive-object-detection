This is a detailed explanation of the module combining the result of `vtd_bbox.py` and object detection module.

## Inputs
3 input ports will be defined. 
- The raw image propagated through the `vtd_bbox` module
- The information of the heatmap bunding box from the `vtd_bbox`
- The information of the Object bounding boxes and labels from object detection module

## Outputs
The output for this module would be the Image and the bounding boxes of all the objects in the scene, with the visually attended object being highlighted.

## Approach
- IoU 

    To decide which object is being visually attended, we will apply the Intersection over Union (IoU) metric. In simple words, we will compare the heatmap bounding box extracted from the `vtd_bbox` with each of the bounding boxes extracted from the object detection module and choose the one with the highest IoU value as the visually attended object.

    <img src="Img/IoU.png" width=200>

- Object detection output

    Theoutput of the object detection module has the form:
    ```
    -1 0.000000 1608649888.545794 (128.0 286.0 170.0 402.0 0.877657055854797363281 pringles (orangecup mustard pear glass pringles)) (236.0 326.0 273.0 393.0 0.919657945632934570312 glass (orangecup mustard pear glass pringles)) (312.0 423.0 354.0 460.0 0.812700802087783813477 pear (orangecup mustard pear glass pringles)) (421.0 341.0 469.0 439.0 0.747145771980285644531 mustard (orangecup mustard pear glass pringles)) (486.0 359.0 508.0 393.0 0.747145771980285644531 mustard (orangecup mustard pear glass pringles))

    ```
    which is a string containing some values separated by spaces.
    1. index
    2. time indicator
    3. time indicators
    4. Xtl
    5. Ytl
    6. Xbr
    7. Ybr
    8. Confidence
    9. Label
    10. (set of all possible labels)
    
    The bounding box values and also the label of each detection needs to be extracted. However, The output of detection is acquired through the YARP port and is not a python list. To handle it, we will recieve it as a bottle and then store the information in a python list. The detailed explanation of the related code can be found below:

    - A yarp bottle is created. The `clear` is to prevent errors and have empty bottle each time.  And read the detection data from a yarp port.

        ```
        obj_det_data = yarp.Bottle()
        obj_det_data.clear()
        obj_det_data = self.in_port_objdet_data.read()
        ```
    - An empty list `predictions` will append the information of the bottle as a list. The information size recieved from the port is not known for us and we use `obj_det_data.size()` to loop over all the info. Considering the output shape of the object detection (as mentioned before), and the fact that the module will recieve them one by one, first each detection is seperated and then for each detection the bounding box info and the label will get extracted. These information is recieved in a dictionary and will be reached easily when needed.
        - NOTE: The bounding box infrmation are of the type `Float64`, and the if statement is to prevent errors and make sure we have recieved onformation.
        ```
        predictions= []
        if obj_det_data is not None:  
            for i in range(0, obj_det_data.size()):  
                dets = obj_det_data.get(i).asList() 
                if dets.get(0).isFloat64():  
                    bbox = [dets.get(0).asFloat64(), dets.get(1).asFloat64(), dets.get(2).asFloat64(),  
                            dets.get(3).asFloat64()] # bbox format: [tl_x, tl_y, br_x, br_y]
                    cls = dets.get(5).asString() # label of i-th detection

                    detection_dict = { 
                        'bbox': bbox,  
                        'class': cls  } 
                    predictions.append(detection_dict)
        ```

An IoU function is defined and then called whenever needed. 

```
 # IoU
    def iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Intersection
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Union
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        unionArea = boxAArea + boxBArea - interArea

        iou = interArea / float(unionArea)

        # return the intersection over union value
        return iou
```
- NOTE: Not using `self` as an input will result in the error. To be more clear, the function is defined inside a class and when called like `self.iou` it will automatically include the self as an output. Therefore, there will be 3 inputs for the method and if you don't use self as an input when defining it there will be an error relatyed to the number of inputs included.

## Exceptions
There are some exceptions that may affect the result. 
1. Object detection module fails

    If Object detection module does not provide any output the module cannot proceed properly. To cover this exception an if statement is used.
    ```
    if obj_det_data is not None:  
    ...
    else:
    print("No information recieved from object detection module")
    ```
2. IoU is 0

    If the heatmap bounding box has no overlap with any of the objects in the scene, no object would be selected. To overcome this problem another comparison based on the distance of the bounding box center is applied. With this comparison, if no object is selected, the heatmap boundingbox center is compared with the center of each object's bounding box andd the one with the minimum distance is chosen as the visually attended object.

    ```
    else:
    min_distance = 1000000
    center_hm_bbox = ((hm_bbox_data_list[0] + hm_bbox_data_list[2])/2 , (hm_bbox_data_list[1] + hm_bbox_data_list[3])/2)    
    for pred in predictions:
        obj_label = pred.get("class")
        obj_det_bbox = pred.get("bbox")
        obj_det_bbox_float32 = np.array(obj_det_bbox, dtype=np.float32).tolist()
        center_obj_det_bbox = ((obj_det_bbox_float32[0] + obj_det_bbox_float32[2])/2 , (obj_det_bbox_float32[1] + obj_det_bbox_float32[3])/2)
        distance = self.dist(center_hm_bbox, center_obj_det_bbox)
        if distance < min_distance:
            min_distance = distance
            selected_obj_label = obj_label
            selected_obj_bbox = obj_det_bbox_float32  
    print("The visually attended object selected by DISTANCE is", selected_obj_label)                  
    selected_label = cv2.putText(wallpaper, selected_obj_label, (int(selected_obj_bbox[0]), int(selected_obj_bbox[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 2)
    selected_obj = cv2.rectangle(selected_label, (int(selected_obj_bbox[0]), int(selected_obj_bbox[1])), (int(selected_obj_bbox[2]), int(selected_obj_bbox[3])), (0, 255, 0), 5)
    # Output to yarp port
    self.out_buf_detection_array[:, :] = selected_obj
    self.out_port_detection_image.write( self.out_buf_detection_image)
    ```