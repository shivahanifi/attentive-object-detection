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
from PIL import Image
from config_vt import *



# Initialize YARP
yarp.Network.init()

class AttentiveObjectDetection(yarp.RFModule):
    def configure(self, rf):  
        
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

        # Output port for selected bbox data
        self.out_port_bbox_data = yarp.Port()
        self.out_port_bbox_data.open('/aod/bbox:o')
        print('{:s} opened'.format('/aod/bbox:o'))

        # Output port for image and bboxes
        self.out_port_detection_image = yarp.Port()
        self.out_port_detection_image.open('/aod/detect:o')
        self.out_buf_detection_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_detection_image = yarp.ImageRgb()
        self.out_buf_detection_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_detection_image.setExternal(self.out_buf_detection_array.data, self.out_buf_detection_array.shape[1],
                                             self.out_buf_detection_array.shape[0])
        print('{:s} opened'.format('/aod/detect:o'))

        # Propag output image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/aod/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/aod/propag:o'))

        self.no_hm_bbox_count = 0
        self.old_predictions = []
        self.obj_det_memory = None

        return True
    
    # Respond to a message
    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            # command: quit
            #self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        return True
    
    # Clean the ports
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
        
        return interArea / float(unionArea)

    # Distance
    def dist(self, p, q):
        if len(p) != len(q):
            raise ValueError("Points must have the same number of dimensions")
        return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(len(p))))
    
    def updateModule(self):
        # Recieve the inpu image
        received_image = self.in_port_scene_image.read()
        if received_image is not None:
            self.in_buf_scene_image.copy(received_image)
            assert self.in_buf_scene_array.__array_interface__['data'][0] == self.in_buf_scene_image.getRawImage().__int__()   
            frame_raw = Image.fromarray(self.in_buf_scene_array) 

            # Recieve object detection data
            obj_det_data = yarp.Bottle()
            obj_det_data.clear()
            obj_det_data = self.in_port_objdet_data.read()
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
                if len(predictions) == 0:
                   predictions = list(self.old_predictions)
                else:
                   self.old_predictions = list(predictions)
            
                # Recieve heatmap bbox data
                hm_bbox_data = yarp.Bottle()
                hm_bbox_data.clear()
                hm_bbox_data = self.in_port_hm_bbox_data.read(False)
                self.hm_bbox_data_memory = hm_bbox_data
                hm_bbox_data_list = []

                if hm_bbox_data is not None:
                    for i in range(hm_bbox_data.size()):
                        hm_bbox_data_list.append(hm_bbox_data.get(i).asFloat32())
                    # Selection & Visualization       
                    max_iou = 0
                    selected_obj_label = None
                    wallpaper = np.asarray(frame_raw)
                    for pred in predictions:
                        obj_label = pred.get("class")
                        obj_det_bbox = pred.get("bbox")
                        obj_det_bbox_float32 = np.array(obj_det_bbox, dtype=np.float32).tolist()
                        top_left = (int(obj_det_bbox[0]), int(obj_det_bbox[1]))
                        bottom_right = (int(obj_det_bbox[2]), int(obj_det_bbox[3]))
                        label_draw = cv2.putText(wallpaper, obj_label, (top_left[0], top_left[1]-20), 

                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, 2)
                        bbox_draw = cv2.rectangle(label_draw, top_left, bottom_right, (0, 0, 255), 2)
                        wallpaper = bbox_draw         
                        iou_value = self.iou(hm_bbox_data_list,obj_det_bbox_float32)
                        if iou_value > max_iou:
                            selected_obj_label = obj_label               
                            selected_obj_bbox = obj_det_bbox
                            max_iou = iou_value


                    if selected_obj_label is not None:
                        print("The visually attended object selected by IoU is", selected_obj_label)
                        selected_label = cv2.putText(wallpaper, selected_obj_label, (int(selected_obj_bbox[0]),
                                                    int(selected_obj_bbox[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.7, (0, 255, 0), 2, 2)
                        selected_obj = cv2.rectangle(selected_label, (int(selected_obj_bbox[0]), 
                                                    int(selected_obj_bbox[1])), (int(selected_obj_bbox[2]), 
                                                    int(selected_obj_bbox[3])), (0, 255, 0), 3)
                        
                        # Output to yarp port- selected bbox data
                        selected_bbox_data = yarp.Bottle()
                        selected_bbox_data.addFloat32(selected_obj_bbox[0])
                        selected_bbox_data.addFloat32(selected_obj_bbox[1])
                        selected_bbox_data.addFloat32(selected_obj_bbox[2])
                        selected_bbox_data.addFloat32(selected_obj_bbox[3])
                        selected_bbox_data.addString(selected_obj_label)
                        self.out_port_bbox_data.write(selected_bbox_data)

                        # Output to yarp port- Image
                        self.out_buf_detection_array[:, :] = selected_obj
                        self.out_port_detection_image.write( self.out_buf_detection_image)
                    else:
                        print('here')
                        min_distance = 10000000
                        center_hm_bbox = ((hm_bbox_data_list[0] + hm_bbox_data_list[2])/2 , 
                                        (hm_bbox_data_list[1] + hm_bbox_data_list[3])/2)
                        if True:
                            self.old_predictions = predictions
                            for pred in predictions:
                                print('Inside predictions')
                                obj_label = pred.get("class")
                                obj_det_bbox = pred.get("bbox")
                                obj_det_bbox_float32 = np.array(obj_det_bbox, dtype=np.float32).tolist()
                                print(obj_det_bbox_float32)
                                if len(obj_det_bbox_float32):
                                    center_obj_det_bbox = ((obj_det_bbox_float32[0] + obj_det_bbox_float32[2])/2 , 
                                                    (obj_det_bbox_float32[1] + obj_det_bbox_float32[3])/2)
                                    distance = self.dist(center_hm_bbox, center_obj_det_bbox)
                                    print('detected distance is: ',distance)
                                    
                                    if distance < min_distance:
                                        min_distance = distance
                                        selected_obj_label = obj_label
                                        selected_obj_bbox = obj_det_bbox_float32

#                        else:
#                            print('No object detected, trying the memory object')
#                            predictions = self.predictions
#                            obj_det_bbox_float32 = np.array(obj_det_bbox, dtype=np.float32).tolist()
#                            center_obj_det_bbox = ((obj_det_bbox_float32[0] + obj_det_bbox_float32[2])/2 ,
#                                                    (obj_det_bbox_float32[1] + obj_det_bbox_float32[3])/2)
#                            distance = self.dist(center_hm_bbox, center_obj_det_bbox)
#                            print('ditance is: ', distance)
#                            if distance < min_distance:
#                                min_distance = distance
#                                selected_obj_label = obj_label
#                                selected_obj_bbox = obj_det_bbox_float32  
                        print("The visually attended object selected by DISTANCE is", selected_obj_label)                  
                        selected_label = cv2.putText(wallpaper, selected_obj_label, (int(selected_obj_bbox[0]),
                                                    int(selected_obj_bbox[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.7, (0, 255, 0), 2, 2)
                        selected_obj = cv2.rectangle(selected_label, (int(selected_obj_bbox[0]), 
                                                    int(selected_obj_bbox[1])), (int(selected_obj_bbox[2]),
                                                    int(selected_obj_bbox[3])), (0, 255, 0), 5)
                        
                                                    
                        # Output to yarp port- selected bbox data
                        selected_bbox_data = yarp.Bottle()
                        selected_bbox_data.addFloat32(selected_obj_bbox[0])
                        selected_bbox_data.addFloat32(selected_obj_bbox[1])
                        selected_bbox_data.addFloat32(selected_obj_bbox[2])
                        selected_bbox_data.addFloat32(selected_obj_bbox[3])
                        selected_bbox_data.addString(selected_obj_label)
                        self.out_port_bbox_data.write(selected_bbox_data)


                        # Output to yarp port
                        self.out_buf_detection_array[:, :] = selected_obj
                        self.out_port_detection_image.write( self.out_buf_detection_image)
                        buffered_output = self.out_buf_detection_image


                        # Copy output to propagation port
                        self.out_buf_propag_image_array = selected_obj
                        self.out_port_propag_image.write(self.out_buf_propag_image)
                else:
                    if False:
                        # Output to yarp port
                        print('No heatmap detected, providing the previous detection')
                        
                        hm_bbox_data = self.hm_bbox_data_memory
                        hm_bbox_data_list = []
                        for i in range(hm_bbox_data.size()):
                            hm_bbox_data_list.append(hm_bbox_data.get(i).asFloat32())
                        # Selection & Visualization       
                        max_iou = 0
                        selected_obj_label = None
                        wallpaper = np.asarray(frame_raw)
                        for pred in predictions:
                            obj_label = pred.get("class")
                            obj_det_bbox = pred.get("bbox")
                            obj_det_bbox_float32 = np.array(obj_det_bbox, dtype=np.float32).tolist()
                            top_left = (int(obj_det_bbox[0]), int(obj_det_bbox[1]))
                            bottom_right = (int(obj_det_bbox[2]), int(obj_det_bbox[3]))
                            label_draw = cv2.putText(wallpaper, obj_label, (top_left[0], top_left[1]-20),

                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, 2)
                            bbox_draw = cv2.rectangle(label_draw, top_left, bottom_right, (0, 0, 255), 2)
                            wallpaper = bbox_draw
                            iou_value = self.iou(hm_bbox_data_list,obj_det_bbox_float32)
                            if iou_value > max_iou:
                                selected_obj_label = obj_label
                                selected_obj_bbox = obj_det_bbox
                                max_iou = iou_value

                        if selected_obj_label is not None:
                            print("The visually attended object selected by IoU is", selected_obj_label)
                            selected_label = cv2.putText(wallpaper, selected_obj_label, (int(selected_obj_bbox[0]),
                                                    int(selected_obj_bbox[1])-20), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.7, (0, 255, 0), 2, 2)
                            selected_obj = cv2.rectangle(selected_label, (int(selected_obj_bbox[0]),
                                                    int(selected_obj_bbox[1])), (int(selected_obj_bbox[2]),
                                                    int(selected_obj_bbox[3])), (0, 255, 0), 3)

                            # Output to yarp port- selected bbox data
                            selected_bbox_data = yarp.Bottle()
                            selected_bbox_data.addFloat32(selected_obj_bbox[0])
                            selected_bbox_data.addFloat32(selected_obj_bbox[1])
                            selected_bbox_data.addFloat32(selected_obj_bbox[2])
                            selected_bbox_data.addFloat32(selected_obj_bbox[3])
                            selected_bbox_data.addString(selected_obj_label)
                            self.out_port_bbox_data.write(selected_bbox_data)

                            # Output to yarp port- Image
                            self.out_buf_detection_array[:, :] = selected_obj
                            self.out_port_detection_image.write( self.out_buf_detection_image)
                        else:
                            min_distance = 1000000000000000000
                            center_hm_bbox = ((hm_bbox_data_list[0] + hm_bbox_data_list[2])/2 ,
                                        (hm_bbox_data_list[1] + hm_bbox_data_list[3])/2)
                            for pred in predictions:
                                obj_label = pred.get("class")
                                obj_det_bbox = pred.get("bbox")
                                obj_det_bbox_float32 = np.array(obj_det_bbox, dtype=np.float32).tolist()
                                center_obj_det_bbox = ((obj_det_bbox_float32[0] + obj_det_bbox_float32[2])/2 ,
                                                    (obj_det_bbox_float32[1] + obj_det_bbox_float32[3])/2)
                            distance = self.dist(center_hm_bbox, center_obj_det_bbox)
                            if distance < min_distance:
                                min_distance = distance
                                selected_obj_label = obj_label
                                selected_obj_bbox = obj_det_bbox_float32
                            print("The visually attended object selected by DISTANCE is", selected_obj_label)
                            selected_label = cv2.putText(wallpaper, selected_obj_label, (int(selected_obj_bbox[0]),
                                                    int(selected_obj_bbox[1])-20), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.7, (0, 255, 0), 2, 2)
                            selected_obj = cv2.rectangle(selected_label, (int(selected_obj_bbox[0]),
                                                    int(selected_obj_bbox[1])), (int(selected_obj_bbox[2]),
                                                    int(selected_obj_bbox[3])), (0, 255, 0), 5)


                            # Output to yarp port- selected bbox data
                            selected_bbox_data = yarp.Bottle()
                            selected_bbox_data.addFloat32(selected_obj_bbox[0])
                            selected_bbox_data.addFloat32(selected_obj_bbox[1])
                            selected_bbox_data.addFloat32(selected_obj_bbox[2])
                            selected_bbox_data.addFloat32(selected_obj_bbox[3])
                            selected_bbox_data.addString(selected_obj_label)
                            self.out_port_bbox_data.write(selected_bbox_data)


                            # Output to yarp port
                            self.out_buf_detection_array[:, :] = selected_obj
                            self.out_port_detection_image.write( self.out_buf_detection_image)
                            buffered_output = self.out_buf_detection_image


                            # Copy output to propagation port
                            self.out_buf_propag_image_array = selected_obj
                            self.out_port_propag_image.write(self.out_buf_propag_image)

                        self.no_heatmap_count +=1
                    else:
                        print('No heatmap detected for more than 3 frames')
                        no_object = cv2.putText(np.asarray(frame_raw), 'Non of the objects visually attended.', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 
                                                 0.7, (255, 0, 0), 2, 2)
                        no_object_array = np.asarray(no_object)
                        self.out_buf_detection_array[:, :] = no_object_array
                        self.out_port_detection_image.write( self.out_buf_detection_image)

                        # Output to yarp port- selected bbox data
                        selected_bbox_data = yarp.Bottle() 
                        selected_bbox_data.addString('None of the objects visually attended.')
                        self.out_port_bbox_data.write(selected_bbox_data)
            else:
                print("No information recieved from object detection module")
        else:
            print('NO INPUT IMAGE.')
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
    
