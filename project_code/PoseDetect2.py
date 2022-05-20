'''
Based off of Ethan Dell's work: https://medium.com/analytics-vidhya/pose-estimation-on-the-raspberry-pi-4-83a02164eb8e 
'''
import os
import cv2
import numpy as np
import sys
import pdb
import time
import math
import pathlib
from threading import Thread
import importlib.util
import datetime
import argparse
import random

if importlib.util.find_spec('tensorflow') is None:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

class PoseDetect2:
    """
    Class that does pose detection using a frame
    """

    def __init__(self, mdl, frame=None):
        self.in_frame = frame
        self.stopped = False
        MODEL_NAME = mdl
        CWD_PATH = os.getcwd()
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)
        self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.output_stride = 32
        self.GRAPH_NAME = 'detect.tflite'
        self.LABELMAP_NAME = 'labelmap.txt'
        self.min_conf_threshold = 0.5
        self.imW = 640
        self.imH = 480

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5
        self.detection_frame = None
        self.keypoint_positions_r = None
        self.keypoint_positions_l = None
        self.drop_pts_r = None
        self.drop_pts_l = None

    def start(self):
        Thread(target=self.detect, args=()).start()
        return self

    def mod(self, a, b):
        """find a % b"""
        floored = np.floor_divide(a, b)
        return np.subtract(a, np.multiply(floored, b))

    def sigmoid(self, x):
        """apply sigmoid actiation to numpy array"""
        return 1/ (1 + np.exp(-x))
        
    def sigmoid_and_argmax2d(self, inputs, threshold):
        """return y,x coordinates from heatmap"""
        #v1 is 9x9x17 heatmap
        v1 = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        height = v1.shape[0]
        width = v1.shape[1]
        depth = v1.shape[2]
        reshaped = np.reshape(v1, [height * width, depth])
        reshaped = self.sigmoid(reshaped)
        #apply threshold
        reshaped = (reshaped > threshold) * reshaped
        coords = np.argmax(reshaped, axis=0)
        yCoords = np.round(np.expand_dims(np.divide(coords, width), 1)) 
        xCoords = np.expand_dims(self.mod(coords, width), 1) 
        return np.concatenate([yCoords, xCoords], 1)

    def get_offset_point(self, y, x, offsets, keypoint, num_key_points):
        """get offset vector from coordinate"""
        y_off = offsets[y,x, keypoint]
        x_off = offsets[y,x, keypoint+num_key_points]
        return np.array([y_off, x_off])
        

    def get_offsets(self, output_details, coords, num_key_points=17):
        """get offset vectors from all coordinates"""
        offsets = self.interpreter.get_tensor(output_details[1]['index'])[0]
        offset_vectors = np.array([]).reshape(-1,2)
        for i in range(len(coords)):
            heatmap_y = int(coords[i][0])
            heatmap_x = int(coords[i][1])
            #make sure indices aren't out of range
            if heatmap_y >8:
                heatmap_y = heatmap_y -1
            if heatmap_x > 8:
                heatmap_x = heatmap_x -1
            offset_vectors = np.vstack((offset_vectors, self.get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)))  
        return offset_vectors

    def draw_lines(self, keypoints, image, bad_pts):
        """connect important body part keypoints with lines"""
        #color = (255, 0, 0)
        color = (0, 255, 0)
        thickness = 2
        #refernce for keypoint indexing: https://www.tensorflow.org/lite/models/pose_estimation/overview
        body_map = [[5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
        for map_pair in body_map:
            #print(f'Map pair {map_pair}')
            if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
                continue
            start_pos = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
            end_pos = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
            image = cv2.line(image, start_pos, end_pos, color, thickness)
        return image

    def detect(self):
        while self.in_frame is not None:
            frame = self.in_frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height,width,channels = frame_rgb.shape
            half_width = int(width/2)
            frameR = frame_rgb[0:height, 0:half_width]
            frameL = frame_rgb[0:height, half_width:width]

            right = True
            detection_frame_r = None
            detection_frame_l = None
            for frame in [frameR,frameL]:
                frame_resized = cv2.resize(frame, (self.width, self.height))
                input_data = np.expand_dims(frame_resized, axis=0)
                
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                if self.floating_model:
                    input_data = (np.float32(input_data) - self.input_mean) / self.input_std
                # print(frame_resized.shape)
                self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'],input_data)
                self.interpreter.invoke()
                #get y,x positions from heatmap
                coords = self.sigmoid_and_argmax2d(self.output_details, self.min_conf_threshold)
                #keep track of keypoints that don't meet threshold
                drop_pts = list(np.unique(np.where(coords ==0)[0]))
                
                if right:
                    self.drop_pts_r = drop_pts
                else:
                    self.drop_pts_l = drop_pts
                #get offets from postions
                offset_vectors = self.get_offsets(self.output_details, coords)
                keypoint_positions = coords * self.output_stride + offset_vectors
                #use stide to get coordinates in image coordinates
                if right:
                    self.keypoint_positions_r =  keypoint_positions
                else:
                    self.keypoint_positions_l = keypoint_positions


                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(keypoint_positions)):
                    #don't draw low confidence poi nts
                    if i in drop_pts:
                        continue
                    # Center coordinates
                    x = int(keypoint_positions[i][1])
                    y = int(keypoint_positions[i][0])
                    center_coordinates = (x, y)
                    radius = 2
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.circle(frame_resized, center_coordinates, radius, color, thickness)
                    # if debug:
                    cv2.putText(frame_resized, str(i), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1) # Draw label text
                    # print('detected')
                detection_frame = self.draw_lines(keypoint_positions, frame_resized, drop_pts)

                if right:
                    detection_frame_r = detection_frame
                    right = False
                else:
                    detection_frame_l = detection_frame

            self.detection_frame = cv2.hconcat([detection_frame_l,detection_frame_r])

    def output_frame(self):
        return self.detection_frame