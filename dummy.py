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
from utils import *
import matplotlib
matplotlib.use('Agg')

image_stack = []
stack_lock = threading.Lock()

# Initialize the camera (you may need to adjust the camera index)
cap = cv2.VideoCapture(0)

# Create a flag to signal when to stop capturing frames
stop_capturing = False

displayed_frames = 0

# Function to capture frames from the camera
def capture_frames():
    global stop_capturing
    while not stop_capturing:
        ret, frame = cap.read()
        if ret:
            with stack_lock:
                image_stack.append(frame)

# Create a thread for capturing frames
capture_thread = threading.Thread(target=capture_frames)

# Function to display frames using OpenCV
def display_frames():
    global stop_capturing
    global displayed_frames
    blink_count = 0
    ptime = 0
    ptime_blink = 0
    fps = 30.00
    bpm = 15.00
    running_average = 0.2
    ear_list = []
    running_average_array = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while not stop_capturing:
        
        with stack_lock:
            if not image_stack:
                continue  # Wait if the stack is empty
            image = image_stack.pop()
            image_stack.clear()

        # Process the image here (replace with your processing logic)
        if image is not None:

            # Get the size of the frame
            height, width, channels = image.shape

            ######### FPS counter  logic ##########
            displayed_frames += 1
            if displayed_frames % 30 == 0:
                current_time_seconds = time.time()
                fps = 30 / (current_time_seconds - ptime)
                ptime = current_time_seconds
            #######################################

            ############### Get eye aspect ratio ##############
            # convert the frame to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            faces = detector(gray)
            # loop over available faces
            for face in faces:
                left_eye, right_eye = get_eye_landmarks(gray, face, predictor)
                ear = get_ear(left_eye, right_eye)
                ear_list.append(ear)
                running_average = calc_running_avg(running_average, ear)
                running_average_array.append(running_average)
                break # we only need one face
            ####################################################

            
            ######### BPM counter  logic ##########
            if(len(ear_list) > 3):
                if (ear_list[-1] < running_average_array[-2]*0.8) and not (ear_list[-2] < running_average_array[-3]*0.8):
                    blink_count += 1
            current_time_seconds = time.time()
            if current_time_seconds - ptime_blink > 60:
                bpm = blink_count / (current_time_seconds - ptime_blink) * 60
                ptime_blink = current_time_seconds
                blink_count = 0
            #######################################

            if(len(ear_list) > 100):
                ear_list = ear_list[-100:]
                running_average_array = running_average_array[-100:]

            # x = np.linspace(0, len(ear_list)-1, len(ear_list))
            # smooth_y = gaussian_filter1d(ear_list, 1)
            # plt.plot(x, smooth_y, color = 'blue')
            # plt.plot(x, running_average_array, color = 'green')
            # plot_filename = 'plot.png'
            # plt.savefig(plot_filename)
            # plt.clf()
            # plot = cv2.imread(plot_filename)
            # plot = cv2.resize(plot, (width, height))

            cv2.putText(image, f"FPS: {round(fps, 2)}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Blinks per minute: {round(bpm)}", (370, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # combined_image = cv2.hconcat([image, plot])
            cv2.imshow('Camera Feed', image)
            cv2.waitKey(1)

# Create a thread for displaying frames
display_thread = threading.Thread(target=display_frames)

# Start both threads
capture_thread.start()
display_thread.start()

# Wait for user input to stop capturing
input("Press Enter to stop capturing...")

# Set the flag to stop capturing
stop_capturing = True

# Wait for both threads to finish
capture_thread.join()
display_thread.join()

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()