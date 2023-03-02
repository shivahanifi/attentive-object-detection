import yarp
import cv2
import math
import PIL
import sys
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config_vt import *



# Initialize YARP
yarp.Network.init()

class AttentiveObjectDetection(yarp.RFModule):
    def configure(self, rf):
        
        # GPU
        num_gpu = rf.find("num_gpu").asInt32()
        num_gpu_start = rf.find("num_gpu_start").asInt32()
        print('Num GPU: %d, GPU start: %d' % (num_gpu, num_gpu_start))
        init_gpus(num_gpu, num_gpu_start)    
        
        # Command port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/aod/command:i')
        print('{:s} opened'.format('/aod/command:i'))
        self.attach(self.cmd_port)
        
        # Input port and buffer for rgb image
        # Create the port and name it
        self.in_port_scene_image = yarp.BufferedPortImageRgb()
        self.in_port_scene_image.open('/aod/image:i')
        # Create numpy array to receive the image 
        self.in_buf_scene_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_scene_image = yarp.ImageRgb()
        self.in_buf_scene_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        # Wrap YARP image around the array
        self.in_buf_scene_image.setExternal(self.in_buf_scene_array.data, self.in_buf_scene_array.shape[1],
                                            self.in_buf_scene_array.shape[0])
        print('{:s} opened'.format('/aod/image:i'))
                
        # Input port for heatmap bbox data
        self.in_port_hm_bbox_data = yarp.BufferedPortBottle()
        self.in_port_hm_bbox_data.open('/aod/hmbbox:i')
        print('{:s} opened'.format('/aod/hmbbox:i'))

        # Input port for object detection data
        self.in_port_objdet_data = yarp.BufferedPortBottle()
        self.in_port_objdet_data.open('/aod/objdet:i')
        print('{:s} opened'.format('/aod/objdet:i'))

        # Output port for image and bboxes
        self.out_port_detection_image = yarp.Port()
        self.out_port_detection_image.open('/aod/detect:o')
        self.out_buf_detection_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_detection_image = yarp.ImageRgb()
        self.out_buf_detection_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_detection_image.setExternal(self.out_buf_thresh_array.data, self.out_buf_thresh_array.shape[1],
                                             self.out_buf_thresh_array.shape[0])
        print('{:s} opened'.format('/aod/detect:o'))

        # Propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/aod/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/aod/propag:o'))

        return True
    
    # Respond to a message
    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            # command: quit
            #self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        return True
    
    def cleanup(self):
        print('Cleanup function')
        self.in_port_scene_image.close()
        self.in_port_hm_bbox_data.close()
        self.in_port_objdet_data.close()
        self.out_port_detection_image.close()
        self.cmd_port.close()
        return True

    # Closes all the ports after execution
    def close(self):
        print('Cleanup function')
        self.in_port_scene_image.close()
        self.in_port_hm_bbox_data.close()
        self.in_port_objdet_data.close()
        self.out_port_detection_image.close()
        self.cmd_port.close()
        return True


    # Called after a quit command (Does nothing)
    def interruptModule(self):
        print('Interrupt function')
        self.in_port_scene_image.close()
        self.in_port_hm_bbox_data.close()
        self.in_port_objdet_data.close()
        self.out_port_detection_image.close()
        self.cmd_port.close()
        return True

    # Desired period between successive calls to updateModule()
    def getPeriod(self):
        return 0.001
    
    # IoU
    def iou(boxA, boxB):
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

    def updateModule(self):

        # Recieve the inpu image
        frame_raw = self.in_port_scene_image.read()
        self.in_port_scene_image.copy(frame_raw)
        self.out_buf_propag_image.copy(frame_raw)
        assert self.in_buf_scene_array.__array_interface__['data'][0] == self.in_buf_scene_image.getRawImage().__int__()   


        # Recieve heatmap bbox data
        hm_bbox_data = self.in_port_hm_bbox_data.read()

        # Recieve object detection data
        obj_det_data = yarp.Bottle()
        obj_det_data.clear()
        obj_det_data = self.in_port_objdet_data.read()
        predictions= []
        if obj_det_data is not None:  
            for i in range(0, obj_det_data.size()):  
                dets = obj_det_data.get(i).asList() 
                if dets.get(0).isDouble():  
                    bbox = [dets.get(0).asDouble(), dets.get(1).asDouble(), dets.get(2).asDouble(),  
                            dets.get(3).asDouble()] # bbox format: [tl_x, tl_y, br_x, br_y]
                    cls = dets.get(5).asString() # label of i-th detection

                    detection_dict = { 
                        'bbox': bbox,  
                        'class': cls  } 
                    predictions.append(detection_dict)

        # Selection & Visualization       
        max_iou = 0
        wallpaper = frame_raw
        for pred in predictions:
            obj_det_bbox = detection_dict.get("bbox")
            bbox_draw = cv2.rectangle(wallpaper, obj_det_bbox[0:2], obj_det_bbox[2:4], (0, 0, 255), 3)
            wallpaper = bbox_draw            
            iou_value = self.iou(hm_bbox_data, obj_det_bbox)
            if iou_value > max_iou:
                selected_obj_label = detection_dict.get("class")
                selected_obj_bbox = obj_det_bbox
                max_iou = iou_value
        
        print("The visually attended object is", selected_obj_label)
        selected_obj = cv2.rectangle(wallpaper, selected_obj_bbox[0:2], selected_obj_bbox[2:4], (0, 255, 0), 5)

        # Output
        self.out_buf_detection_array[:, :] = selected_obj
        self.out_port_detection_image.write( self.out_buf_detection_image)        

        return True                  

if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("AttentiveObjectDetection")
    rf.setDefaultConfigFile('../app/config/.ini')
    rf.configure(sys.argv)

    # Run module
    manager = AttentiveObjectDetection()
    manager.runModule(rf)
    