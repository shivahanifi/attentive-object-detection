# Computing distance using depth information

import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import struct

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


# camera real sense d415
ICUB_CRIS_CAM_INTRINSIC = np.zeros(shape=(3, 3), dtype=np.float64)
ICUB_CRIS_CAM_INTRINSIC[0, 0] = 618.071  # fx
ICUB_CRIS_CAM_INTRINSIC[0, 2] = 305.902  # cx
ICUB_CRIS_CAM_INTRINSIC[1, 1] = 617.783  # fy
ICUB_CRIS_CAM_INTRINSIC[1, 2] = 246.352  # cy
ICUB_CRIS_CAM_INTRINSIC[2, 2] = 1.0

# def extract_depth_from_float_file(file_path, image_width, point):
#     # Calculate the byte offset in the file based on the specified (x, y) coordinates
#     # Assuming each depth value is represented by a 32-bit (4-byte) floating-point number
#     byte_offset = int((point[1] * image_width + point[0]) * 4)

#     # Open the .float file in binary mode
#     with open(file_path, 'rb') as file:
#         # Seek to the appropriate byte offset
#         file.seek(byte_offset)

#         # Read the floating-point depth value from the file
#         depth_bytes = file.read(4)  # Assuming each depth value is 4 bytes (32 bits)

#         # Interpret the read bytes as a floating-point number
#         depth_value = struct.unpack('f', depth_bytes)[0]
#         print('depth value: ', depth_value)

#     return depth_value

def read_depth(filename):
    with open(filename, 'rb') as f:
        height = f.read(8)
        height = int.from_bytes(height, "little")
        assert height == IMAGE_HEIGHT
        
        width = f.read(8)
        width = int.from_bytes(width, "little")
        assert width == IMAGE_WIDTH

        depth_img = []
        while (True):
            depthval_b = f.read(4)      # binary, little endian
            if not depthval_b:
                break
            depthval_m = struct.unpack("<f", depthval_b)    # depth val as meters
            depth_img.append(depthval_m)
        assert len(depth_img) == height * width

    depth_img = np.array(depth_img, dtype=np.float32).reshape(height, width)

    return depth_img

def get_mean_depth_over_area(depth_img, pixel, range):

    vertical_range = np.zeros(2)
    vertical_range[0] = pixel[1] - round(range/2) if pixel[1] - round(range/2) > 0 else 0
    vertical_range[1] = pixel[1] + round(range/2) if pixel[1] + round(range/2) < IMAGE_HEIGHT else IMAGE_HEIGHT

    horizontal_range = np.zeros(2)
    horizontal_range[0] = pixel[0] - round(range/2) if pixel[0] - round(range/2) > 0 else 0
    horizontal_range[1] = pixel[0] + round(range/2) if pixel[0] + round(range/2) < IMAGE_WIDTH else IMAGE_WIDTH

    vertical_range = vertical_range.astype(int)
    horizontal_range = horizontal_range.astype(int)

    depth = []
    for hpix in np.arange(horizontal_range[0], horizontal_range[1]):
        for vpix in np.arange(vertical_range[0], vertical_range[1]):
            depth.append(depth_img[vpix, hpix])

    mean_depth = np.mean(depth)

    return mean_depth

def from_pixels_to_ccs(point_pixels, depth, cam_intrinsic):
    point_ccs = np.zeros(3)

    point_ccs[2] = depth                                                                     # z
    point_ccs[0] = (point_pixels[0] - cam_intrinsic[0, 2])*point_ccs[2]/cam_intrinsic[0, 0]  # x
    point_ccs[1] = (point_pixels[1] - cam_intrinsic[1, 2])*point_ccs[2]/cam_intrinsic[1, 1]  # y

    return point_ccs


initial_data_path = '/media/suka/My Passport/Humanoids23/Humanoids_performance/epoch10_subsampled_data'
distance_info = []
participants = sorted([f for f in os.listdir(initial_data_path) if os.path.isdir(os.path.join(initial_data_path, f))])
#participants = participants[:1:]
for participant in participants:
    participant_path = os.path.join(initial_data_path, participant)
    sessions = sorted([f for f in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, f))])
    #sessions = sessions[:1:]
    for session in sessions:
        distance_info_session = []
        session_path = os.path.join(participant_path, session)
        settings = sorted([f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))])
        #settings = settings[:1:]
        for setting in settings:
            setting_path = os.path.join(session_path, setting)
            objects = sorted([f for f in os.listdir(setting_path) if os.path.isdir(os.path.join(setting_path, f))])
            #objects = objects[:1:]
            for object in objects:
                object_path = os.path.join(setting_path, object)
                distance_info_object = []
                pred_bbox_centers = []
                distance_info_object_path = os.path.join(object_path, 'distance_info.txt')
                files = os.listdir(object_path)
                for file in files:
                    if '.xml' in file:
                        annotation_path = os.path.join(object_path, file)
                        tree = ET.parse(annotation_path)
                        root = tree.getroot()
                        for obj in root.findall('object'):
                            if obj.find('name').text == object:
                                obj_info = {
                                    'name': obj.find('name').text,
                                    'bndbox': {
                                        'xmin': float(obj.find('bndbox/xmin').text),
                                        'ymin': float(obj.find('bndbox/ymin').text),
                                        'xmax': float(obj.find('bndbox/xmax').text),
                                        'ymax': float(obj.find('bndbox/ymax').text)
                                    }
                                }
                        gt_bbox = [obj_info['bndbox']['xmin'], obj_info['bndbox']['ymin'], obj_info['bndbox']['xmax'], obj_info['bndbox']['ymax']]
                        gt_bbox_center = [(gt_bbox[0] + gt_bbox[2])/2, (gt_bbox[1] + gt_bbox[3])/2]
                    elif 'bbox' in file:
                        file_path = os.path.join(object_path, file)
                        pred_bboxes_info = os.listdir(file_path)
                        for pred_info in pred_bboxes_info:
                            if 'data' in pred_info:
                                pred_bboxes_path = os.path.join(file_path, pred_info)
                                with open(pred_bboxes_path, 'r') as predictions:
                                    line_counter = 0
                                    for line in predictions:                                        
                                        tokens = line.split()
                                        if tokens[3] != '"None':
                                            bounding_box_values = tokens[3:7]
                                            bounding_box_values = [float(value) for value in bounding_box_values]
                                            pred_bbox_center = [(bounding_box_values[0] + bounding_box_values[2])/2, (bounding_box_values[1] + bounding_box_values[3])/2]
                                            pred_bbox_centers.append(pred_bbox_center) 
                                            line_counter += 1                                           
                                        # else:
                                        #    pred_bbox_centers.append()                                         
                                        
                    elif 'depth' in file:
                        depth_files_path = os.path.join(object_path, file)
                        depth_files = sorted([os.path.join(depth_files_path,f) for f in os.listdir(depth_files_path) if 'log' not in f])
                        depth_files_subsampled = depth_files[::2] 
                gt_bbox_center_css_list =[]  
                pred_bbox_center_css_list = []
                for i in range(len(pred_bbox_centers)):
                    depth_info = depth_files_subsampled[i]
                    # with open(depth_info, 'rb') as file:
                    #     data = np.fromfile(file, dtype=np.float32)
                    # # Reshape the data into a 2D depth map
                    # depth_map = np.reshape(data, (640, 480))
                    # print('depth_map is: ', depth_map)
                    depth_image = read_depth(depth_info)
                    gt_bbox_center_css = from_pixels_to_ccs(gt_bbox_center, get_mean_depth_over_area(depth_image, gt_bbox_center, 20), ICUB_CRIS_CAM_INTRINSIC)
                    gt_bbox_center_css_list.append(gt_bbox_center_css)
                    #print('gt_bbox_center_css: ', gt_bbox_center_css)
                    pred_bbox_center_css = from_pixels_to_ccs(pred_bbox_centers[i], get_mean_depth_over_area(depth_image, pred_bbox_center, 20), ICUB_CRIS_CAM_INTRINSIC)
                    pred_bbox_center_css_list.append(pred_bbox_center_css)
                    #print('pred_bbox_center_css', pred_bbox_center_css)
                    distance = np.linalg.norm(gt_bbox_center_css - pred_bbox_center_css)
                    #print('distance', distance)
                    distance_info_object.append(distance)
                    distance_info_session.append(distance)
                    distance_info.append(distance)
                mean_distance_object = np.mean(distance_info_object)
                std_distance_object = np.std(distance_info_object)
                #print("For the participant ", participant, "\n session ", session, "\n setting ", setting, "\n object ", object, "\n the mean for distance is: ", mean_distance_object, "\n the std for distance is: ", std_distance_object)
                with open(distance_info_object_path, mode='w') as txt:
                    for j in range(len(distance_info_object)):
                        sentence_to_write = str(j) + "\n" + "gt_bbox_center_css: " + str(gt_bbox_center_css_list[j]) + "\n" + "pred_bbox_center_css: " + str(pred_bbox_center_css_list[j]) + "\n"+ "distance: " + str(distance_info_object[j]) + "\n"
                        txt.write(sentence_to_write)
                    final_sentence_to_wrie = "\n" + "\n" + "mean_distance_object: " + str(mean_distance_object) + "\n" + "std_distance_object: " + str(std_distance_object)
        mean_distance_session = np.mean(distance_info_session)
        std_distance_session = np.std(distance_info_session)
        distance_info_session_path = os.path.join(session_path, 'distance_info_session.txt')
        with open(distance_info_session_path, mode='w') as txt:
            distance_session_sentence = "For " + participant + " session " + session + "\n mean distance is: " + str(mean_distance_session) + "\n" + "std is: " + str(std_distance_session)
            txt.write(distance_session_sentence)
        print("For the participant ", participant, "\n session ", session, "\n the mean for distance is: ", mean_distance_session, "\n the std for distance is: ", std_distance_session)
overall_distance_mean = np.mean(distance_info)     
overall_distance_std = np.std(distance_info) 
distance_info_path = os.path.join(initial_data_path, 'overall_distance_info.txt')
#print("Overall mean for distance is: ", overall_distance_mean, "std is: ", overall_distance_std)  
with open(distance_info_path, mode='w') as txt:
    distances_sentence = "Overall mean for distance is: " +  str(overall_distance_mean) + "\n" + "std is: " + str(overall_distance_std)
    txt.write(distances_sentence)
             