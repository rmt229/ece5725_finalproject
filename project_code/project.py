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
import pygame.mixer as pymix


import board
import neopixel
import RPi.GPIO as GPIO

from VideoGet import VideoGet
from VideoShow import VideoShow
from PoseDetect import PoseDetect
from PoseDetect2 import PoseDetect2
from MotionDetect import MotionDetect

pixel_pin = board.D18

# The number of NeoPixels
num_pixels = 60
GPIO.setmode(GPIO.BCM)
GPIO.setup(6, GPIO.OUT)

GPIO.setup(26,GPIO.OUT)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm=GPIO.PWM(26,50) #50hz is fine, don't change this
pwm.start(0)

ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    pixel_pin, num_pixels, brightness=0.1, auto_write=False, pixel_order=ORDER
)
def wheel(pos):
    # Input a value 0 to 255 to get a color value.
    # The colours are a transition r - g - b - back to r.
    if pos < 0 or pos > 255:
        r = g = b = 0
    elif pos < 85:
        r = int(pos * 3)
        g = int(255 - pos * 3)
        b = 0
    elif pos < 170:
        pos -= 85
        r = int(255 - pos * 3)
        g = 0
        b = int(pos * 3)
    else:
        pos -= 170
        r = 0
        g = int(pos * 3)
        b = int(255 - pos * 3)
    return (r, g, b) if ORDER in (neopixel.RGB, neopixel.GRB) else (r, g, b, 0)


def rainbow_cycle(wait):
    for j in range(255):
        for i in range(num_pixels):
            pixel_index = (i * 256 // num_pixels) + j
            pixels[i] = wheel(pixel_index & 255)
        pixels.show()
        time.sleep(wait)


GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)

loser = False
winner = False

def win_callback(channel):
    winner = True
    print('detected button')
    rainbow_cycle(0.001)
    time.sleep(5)
    cv2.destroyAllWindows()
    GPIO.cleanup()
    exit()

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--num_player', help='Number of players',
                    default=1)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected keypoints (specify between 0 and 1).',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')


args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
imW = 640
imH = 480
use_TPU = args.edgetpu
num_players = args.num_player

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)

# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()


if __name__ == "__main__":
    #flag for debugging
    debug = True 
    state = "g"
    GPIO.add_event_detect(27, GPIO.FALLING, callback=win_callback, bouncetime=300)
    person_moved = False
    left_moved = False
    right_moved = False
    try:
        f = []
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()
        # videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        videostream = VideoGet(0).start()
        detector1 = None
        detector2 = None
        motion_detector = None
        motion_detectorL = None
        motion_detectorR = None
        video_shower = VideoShow(videostream.frame).start()
        if num_players == 1:
            detector1 = PoseDetect(MODEL_NAME,videostream.frame).start()
            detector2 = PoseDetect(MODEL_NAME,videostream.frame).start()
            motion_detector = MotionDetect().start()
        else:
            detector1 = PoseDetect2(MODEL_NAME,videostream.frame).start()
            detector2 = PoseDetect2(MODEL_NAME,videostream.frame).start()
            motion_detectorL = MotionDetect().start()
            motion_detectorR = MotionDetect().start()
        pwm.ChangeDutyCycle(11)
            # motion_detector = MotionDetect2().start()
        while not winner and not loser:

            if state == "g" and not winner:
                # print("g")
                GPIO.output(6,1)
                time_g = random.uniform(1,7)
                pixels.fill((0, 255, 0))
                pixels.show()
                t = time.time()
                while time.time() - t < time_g and not winner and not loser:
                    tmp = 1
                state = "r"
                pixels.fill((255, 0, 0))
                pixels.show()

            
            if state == "r" and not winner and not loser:
                
                GPIO.output(6,0)
                time_r = random.uniform(3,10)
                t = time.time()

                while time.time() - t < time_r and not winner and not loser:
                    t1 = cv2.getTickCount()
                    
                    # # Grab frame from video stream
                    frame1 = videostream.frame
                    time.sleep(.5) # expirment with this
                    frame2 = videostream.frame
                  
                    detector1.in_frame = frame1
                    video_shower.frame = detector1.output_frame()

                    # frame 2

                    detector2.in_frame = frame2
                    video_shower.frame = detector2.output_frame()
                    if num_players==1:
                        motion_detector.kp1 = detector1.output_positions()
                        motion_detector.kp2 = detector2.output_positions()
                        motion_detector.bad_pts1 = detector1.drop_pts
                        motion_detector.bad_pts2 = detector2.drop_pts

                        if motion_detector.did_move and not person_moved:
                            
                            pwm.ChangeDutyCycle(motion_detector.point_laser())
                            person_moved = True
                            loser = True
                            
                    else:
                        motion_detectorL.kp1 = detector1.keypoint_positions_l
                        motion_detectorL.kp2 = detector2.keypoint_positions_l
                        motion_detectorL.kp1 = detector1.keypoint_positions_l
                        motion_detectorL.bad_pts1 = detector2.drop_pts_l
                        motion_detectorL.bad_pts2 = detector2.drop_pts_l

                        motion_detectorR.kp1 = detector1.keypoint_positions_r
                        motion_detectorR.kp2 = detector2.keypoint_positions_r
                        motion_detectorR.kp1 = detector1.keypoint_positions_r
                        motion_detectorR.bad_pts1 = detector2.drop_pts_r
                        motion_detectorR.bad_pts2 = detector2.drop_pts_r

                        if motion_detectorL.did_move and not left_moved:
                            left_moved = True
                           
                            pwm.ChangeDutyCycle(6.5)
                            if motion_detectorR.did_move and not right_moved:
                                right_moved = True    
                                time.sleep(2)
                                
                                pwm.ChangeDutyCycle(8)
                            loser = True
                        elif motion_detectorR.did_move and not right_moved:
                            right_moved = True
                        
                            pwm.ChangeDutyCycle(8)
                            loser = True
                state = "g"
        if winner:
            print('win')
            rainbow_cycle(0.001)
            cv2.destroyAllWindows()
            GPIO.cleanup()
            exit()

        if loser:
            for i in range(5):
                pixels.fill((255, 0, 0))
                pixels.show()
                time.sleep(0.5)
                pixels.fill((0,0,0))
                pixels.show()
                time.sleep(0.5)
          
            print('game over')
            cv2.destroyAllWindows()
            GPIO.cleanup()
            exit()
            
            
    except KeyboardInterrupt:
        # Clean up
        cv2.destroyAllWindows()
    
        print('Stopped video stream.')
    
        GPIO.cleanup()
        exit()
    
