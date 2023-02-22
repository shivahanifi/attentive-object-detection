import yarp
import cv2
import json
import math
import glob
import yarp
import PIL
import sys
import io
import logging
import torch
import argparse, os
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageShow
from scipy.misc import imresize
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from functions.config_vt import *
from functions.utilities_vt import *


# Initialize YARP
yarp.Network.init()

class VisualTargetDetection(yarp.RFModule):
    def configure(self, rf):
        
        # GPU
        num_gpu = rf.find("num_gpu").asInt32()
        num_gpu_start = rf.find("num_gpu_start").asInt32()
        print('Num GPU: %d, GPU start: %d' % (num_gpu, num_gpu_start))
        init_gpus(num_gpu, num_gpu_start)    
        
        # Command port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/vtd/command:i')
        print('{:s} opened'.format('/vtd/command:i'))
        self.attach(self.cmd_port)
        
        # Input port and buffer for rgb image
        # Create the port and name it
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/vtd/image:i')
        # Create numpy array to receive the image 
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        # Wrap YARP image around the array
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1],
                                            self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:i'))
        
        
        # Input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/vtd/data:i')
        print('{:s} opened'.format('/vtd/data:i'))
        
        # Output port for bboxes
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/vtd/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1],
                                             self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:o'))

        # Propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/vtd/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/vtd/propag:o'))

        # Output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/vtd/pred:o')
        print('{:s} opened'.format('/vtd/pred:o'))
        
        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        # To get argumennts needed
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weights', type=str, help='model weights', default='/projects/online-visual-target-detection/model_demo.pt')
        #parser.add_argument('--image_dir', type=str, help='images', default=FRAMES_DIR)
        #parser.add_argument('--head_dir', type=str, help='head bounding boxes', default=JSON_FILES)
        parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='heatmap')
        parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
        self.args = parser.parse_args()

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
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True

    # Closes all the ports after execution
    def close(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True


    # Called after a quit command (Does nothing)
    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True

    # Desired period between successive calls to updateModule()
    def getPeriod(self):
        return 0.001


    def updateModule(self):
    
        received_image = self.in_port_human_image.read()
        self.in_buf_human_image.copy(received_image)
        assert self.in_buf_human_array.__array_interface__['data'][0] == self.in_buf_human_image.getRawImage().__int__()

        # Convert the numpy array to a PIL image
        pil_image = Image.fromarray(self.in_buf_human_array)
        
       
        # To check the input
        #pil_image.save('/projects/test_images/pil_image.png')

        if pil_image:
            received_data = self.in_port_human_data.read()
            if received_data:
                try:
                    poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                
                    if poses:
                        min_x, min_y, max_x, max_y = get_openpose_bbox(poses)

                        column_names = ['left', 'top', 'right', 'bottom']
                        line_to_write = [[min_x, min_y, max_x, max_y]]
                        df = pd.DataFrame(line_to_write, columns=column_names)
    
                        df['left'] -= (df['right']-df['left'])*0.1
                        df['right'] += (df['right']-df['left'])*0.1
                        df['top'] -= (df['bottom']-df['top'])*0.1
                        df['bottom'] += (df['bottom']-df['top'])*0.1


                        # Transforming images
                        def get_transform():
                            transform_list = []
                            transform_list.append(transforms.Resize((input_resolution, input_resolution)))
                            transform_list.append(transforms.ToTensor())
                            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                            return transforms.Compose(transform_list)


                         # set up data transformation
                        test_transforms = get_transform()

                        model = ModelSpatial()
                        model_dict = model.state_dict()
                        pretrained_dict = torch.load(self.args.model_weights)
                        pretrained_dict = pretrained_dict['model']
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)

                        model.cuda()
                        model.train(False)

                        with torch.no_grad():
                            for i in df.index:
                                frame_raw = pil_image
                            
                                width, height = frame_raw.size

                                head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

                                head = frame_raw.crop((head_box)) # head crop

                                head = test_transforms(head) # transform inputs
                                frame = test_transforms(frame_raw)
                                head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                                            resolution=input_resolution).unsqueeze(0)

                                head = head.unsqueeze(0).cuda()
                                frame = frame.unsqueeze(0).cuda()
                                head_channel = head_channel.unsqueeze(0).cuda()

                                # forward pass
                                raw_hm, _, inout = model(frame, head_channel, head)
                                print(raw_hm.shape)

                                # heatmap modulation
                                raw_hm = raw_hm.cpu().detach().numpy()
                                raw_hm_255 = raw_hm * 255
                                raw_hm_sq = raw_hm.squeeze()
                                raw_hm_sq_255 = raw_hm_255.squeeze()
                                inout = inout.cpu().detach().numpy()
                                inout = 1 / (1 + np.exp(-inout))
                                inout = (1 - inout) * 255
                                norm_map = imresize(raw_hm_sq_255, (height, width)) - inout
                                print(norm_map.shape)

                                # Heatmap bbox extraction
                                ret, thresh_hm = cv2.threshold(norm_map, 100, 255, cv2.THRESH_BINARY)
                                print(thresh_hm.shape)
                                print(thresh_hm.dtype)

                                # Check the number of values returned by cv2.findContours()
                                contours_info = cv2.findContours(thresh_hm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if len(contours_info) == 2:
                                    contours, hierarchy = contours_info
                                else:
                                    contours = contours_info[0]

                                print(len(contours))
                                for contour in contours:
                                    x,y,w,h = cv2.boundingRect(contour)
                                    hm_bbox = cv2.rectangle(np.asarray(frame_raw), (x,y), (x+w,y+h), (0,0,255), 2)


                                # Visualization
                                # Draw the raw_frame and the bbox
                                start_point = (int(head_box[0]), int(head_box[1]))
                                end_point = (int(head_box[2]), int(head_box[3]))
                                img_bbox = cv2.rectangle(hm_bbox,start_point,end_point, (0, 255, 0),2)                      
                                
                                # The arrow mode
                                if self.args.vis_mode == 'arrow':
                                    # in-frame gaze
                                    if inout < self.args.out_threshold: 
                                        pred_x, pred_y = evaluation.argmax_pts(raw_hm_sq_255)
                                        norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                                        circs = cv2.circle(img_bbox, (norm_p[0]*width, norm_p[1]*height),  height/50.0, (35, 225, 35), -1)
                                        line = cv2.line(circs, (norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), (255, 0, 0), 2)

                                        line_array =  np.asarray(line)
                                        self.out_buf_human_array[:, :] = line_array
                                        self.out_port_human_image.write(self.out_buf_human_image)

                                # The heatmap mode
                                else:

                                    # Convert the norm_map gray scale image to a 3-channel image with the 'jet' colormap
                                    norm_map = cv2.merge((norm_map,norm_map))
                                    jet_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

                                    # Create an alpha channel with a value of 0.2
                                    alpha = np.ones((norm_map.shape[0], norm_map.shape[1], 1), dtype=np.uint8) * 51

                                    # Stack the jet_map and alpha channels together to create an RGBA image
                                    rgba_map = np.dstack((jet_map, alpha))

                                    # Display both the bbox and heatmap on the image
                                    img_blend_bbox = cv2.addWeighted(rgba_map, 0.2,  np.asarray(img_bbox), 0.8, 0, dtype=cv2.CV_8U)

                                    # Connect to the output port
                                    img_blend_array = np.asarray(img_blend_bbox)
                                    self.out_buf_human_array[:, :] = img_blend_array
                                    self.out_port_human_image.write(self.out_buf_human_image)

                except Exception as err:
                    print("An error occured while extracting the poses from OpenPose data")
                    print("Unexpected error!!! " + str(err))
        return True                  

if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("VisualTargetDetection")
    rf.setDefaultConfigFile('../app/config/.ini')
    rf.configure(sys.argv)

    # Run module
    manager = VisualTargetDetection()
    manager.runModule(rf)
    