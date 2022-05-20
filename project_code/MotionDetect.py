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

class MotionDetect:
    """
    Class that calculates difference in motion
    """

    def __init__(self,kp1=None,bad_pts1=None,kp2=None,bad_pts2=None):
        self.kp1 = kp1
        self.bad_pts1 = bad_pts1
        self.kp2 = kp2
        self.bad_pts2 = bad_pts2

        self.did_move = False
        self.stopped = False


    def start(self):
        Thread(target=self.moved, args=()).start()
        return self
        
    def moved(self):
        low_sensitive = [0,1,2,3,4,5,6,11,12,15,16]
        low_sens_th = 40
        high_sens_th = 20
        labels = ["nose","eyeL","eyeR", "earL","earR","shouldL","shouldR","elbowL","elbowR", "wristL","wristR", "hipL","hipR","kneeL","kneeR","ankleL","ankleR"]
        
        while not self.stopped:
            if self.kp1 is not None and self.kp2 is not None and not self.did_move:
        
                for i in range(17):
        
                    if i in self.bad_pts1 or i in self.bad_pts2:
                        continue
        
                    th = high_sens_th
                    if i in low_sensitive:
                        th = low_sens_th
                    dist = math.sqrt((self.kp1[i][0]-self.kp2[i][0])**2+(self.kp1[i][1]-self.kp2[i][1])**2)
                    if dist > th:
                        self.did_move = True
                        print(labels[i] + " " + str(dist))

    def point_laser(self):
        midpoint = (np.abs(self.kp2[5][0] + self.kp2[6][0])//2)  #between shouldesrs in image coordinates
        print("mdpt", midpoint)
        pwm_output = ((midpoint*0.019+2.5)) #returns the "degree" as pwm input for motor
        if pwm_output > 9.25: pwm_output = 9.25
        if pwm_output < 4.5: pwm_output = 4.5
        print("pwm: ", pwm_output)
        return pwm_output



