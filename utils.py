import cv2
import threading
import time
import dlib
import cv2
import math
import time
import os
from pygame import mixer 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def compute_ear(eye_landmarks):
    # compute the horizontal distance between the left and right eye landmarks
    h_dist = math.sqrt(eye_landmarks[3].x - eye_landmarks[0].x) ** 2 * 2 + 1e-6

    # compute the vertical distance between the top and bottom eye landmarks
    v_dist = math.sqrt((eye_landmarks[1].y - eye_landmarks[5].y) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2) + 1e-6

    # compute the eye aspect ratio
    ear = v_dist / h_dist

    return ear

def calc_running_avg(running_avg, ear):
    if running_avg == 0:
        running_avg = ear
    else: 
        running_avg = running_avg*0.95 + ear * 0.05
    return running_avg

def get_average_ear(ear_list):
    avg_ear = 0
    for ear in ear_list:
        avg_ear += ear
    avg_ear = avg_ear / len(ear_list)
    return avg_ear

def get_eye_landmarks(gray, face, predictor):
    # define the indexes of the facial landmarks for the left and right eye
    LEFT_EYE_START = 36
    LEFT_EYE_END = 42
    RIGHT_EYE_START = 42
    RIGHT_EYE_END = 48
    # predict the facial landmarks for the face
    landmarks = predictor(gray, face)
    # extract the left and right eye landmarks
    left_eye = landmarks.parts()[LEFT_EYE_START:LEFT_EYE_END]
    right_eye = landmarks.parts()[RIGHT_EYE_START:RIGHT_EYE_END]
    return left_eye, right_eye

def get_ear(left_eye, right_eye):
        # compute the eye aspect ratio (EAR) for the left and right eye
        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        # compute the average EAR for both eyes
        ear = (left_ear + right_ear) / 2
        return ear