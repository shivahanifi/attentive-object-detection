import os
import numpy as np
import math
import cv2
import tensorflow as tf
import json
import yarp
from functions.config_vt import *


# Keypoint extraction
def read_openpose_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)

        poses = []
        conf_poses = []
        faces = []
        conf_faces = []

        for arr in loaded["people"]:
            conf_poses.append(arr["pose_keypoints_2d"][2::3]) #seq[start:end:step]
            arr_poses = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
            poses.append(list(zip(arr_poses[::2], arr_poses[1::2]))) #respectively X and Y

            conf_faces.append(arr["face_keypoints_2d"][2::3])
            arr_faces = np.delete(arr["face_keypoints_2d"], slice(2, None, 3))  # remove confidence values from the array
            faces.append(list(zip(arr_faces[::2], arr_faces[1::2])))

    return poses, conf_poses, faces, conf_faces


def read_openpose_data(received_data):
    body = []
    face = []
    if received_data:
        received_data = received_data.get(0).asList()
        for i in range(0, received_data.size()):
            keypoints = received_data.get(i).asList()
            if keypoints:
                body_person = []
                face_person = []
                for y in range(0, keypoints.size()):
                    part = keypoints.get(y).asList()
                    if part:
                        if part.get(0).asString() == "Face":
                            for z in range(1, part.size()):
                                item = part.get(z).asList()
                                face_part = [item.get(0).asFloat64(), item.get(1).asFloat64(), item.get(2).asFloat64()]
                                face_person.append(face_part)
                        else:
                            body_part = [part.get(1).asFloat64(), part.get(2).asFloat64(), part.get(3).asFloat64()]
                            body_person.append(body_part)

                    else:
                        print('Could not extract part')
                if body_person:
                    body.append(body_person)
                if face_person:
                    face.append(face_person)

            else:
                print('Could not extract keypoints') 
    else:
        print('No input data recieved')
    poses, conf_poses = load_many_poses(body)
    faces, conf_faces = load_many_faces(face)

    return poses, conf_poses, faces, conf_faces


def load_many_poses(data):
    poses = []
    confidences = []

    for person in data:
        poses.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return poses, confidences


def load_many_faces(data):
    faces = []
    confidences = []

    for person in data:
        faces.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return faces, confidences


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    if mean_x >= IMAGE_WIDTH:
        mean_x = IMAGE_WIDTH-1
    if mean_x < 0:
        mean_x = 0
    if mean_y >= IMAGE_HEIGHT:
        mean_y = IMAGE_HEIGHT-1
    if mean_y < 0:
        mean_y = 0

    return [mean_x, mean_y]


def joint_set(p):
    return p[0] != 0.0 or p[1] != 0.0


def get_openpose_bbox(pose):

    n_joints_set = [pose[0][joint] for joint in JOINTS_POSE_FACE if joint_set(pose[0][joint])]
    #logging.debug(n_joints_set)
    if n_joints_set:
        centroid = compute_centroid(n_joints_set)

        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.2
        max_x += (max_x - min_x) * 0.2

        width = max_x - min_x

        min_y = centroid[1] - (width/3)*2
        max_y = centroid[1] + (width/3)*2

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        #print("Joint set empty!")
        return None, None, None, None
    
#eGPU    
def init_gpus(num_gpu, num_gpu_start):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if num_gpu > 0:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpus[num_gpu_start:num_gpu_start+num_gpu], 'GPU')
                print(len(gpus), "Physical GPUs found, Set visible devices: ", tf.config.experimental.get_visible_devices('GPU'))
            else:
                tf.config.experimental.set_visible_devices([], 'GPU')
                print("GPU has been disable!!!")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No physical GPU found")

