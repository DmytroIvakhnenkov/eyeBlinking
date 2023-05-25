import dlib
import cv2
import math
import time
import os
from pygame import mixer 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

mixer.init()
mixer.music.load(os.getcwd()+'\\notification.mp3')


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

def get_eye_landmarks(gray, face):
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


# initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# define the indexes of the facial landmarks for the left and right eye
LEFT_EYE_START = 36
LEFT_EYE_END = 42
RIGHT_EYE_START = 42
RIGHT_EYE_END = 48

# initialize the video capture
cap = cv2.VideoCapture(0)

ear_list = []
y_data = []
# Start timing the loop
start_time = time.time()
seconds_passed = 0
minutes_passed = 0
blink_counter = 0
last_min_blinks = 0
blinks_per_min = 20
running_average = 0
running_average_array = []
isClosed, last_frame_isClosed = False, False

while True:
    # read a frame from the video capture
    ret, frame = cap.read()

    # Get the size of the frame
    height, width, channels = frame.shape

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = detector(gray)

    # loop over each face
    for face in faces:

        left_eye, right_eye = get_eye_landmarks(gray, face)

        ear = get_ear(left_eye, right_eye)

        ear_list.append(ear)

        # check if the average EAR is below the threshold (indicating a blink)
        if time.time()-start_time > seconds_passed:
            # how fast
            seconds_passed += 0.1
            avg_ear = get_average_ear(ear_list)
            ear_list = []
            y_data.append(avg_ear)
            isClosed = False
            running_average = calc_running_avg(running_average, avg_ear)
            # check how closed the eyes are
            if avg_ear < (running_average)*0.90:
                isClosed = True
            if not last_frame_isClosed and isClosed:
                blink_counter += 1
            running_average_array.append(running_average)
            last_frame_isClosed = isClosed

        color = (0, 0, 255)

        for dot in left_eye:
            center = (int(dot.x), int(dot.y))
            cv2.circle(frame, center, 2, color, 1)
        for dot in right_eye:
            center = (int(dot.x), int(dot.y))
            cv2.circle(frame, center, 2, color, 1)

    # display the blink counter on the frame
    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if(int((time.time() - start_time)/60) > minutes_passed):
        minutes_passed +=1
        # display the blink counter on the frame
        blinks_per_min = (blink_counter-last_min_blinks)
        last_min_blinks = blink_counter
        if(blinks_per_min < 20):
            mixer.music.play()

    cv2.putText(frame, f"Blinks per minute: {blinks_per_min}", (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Press \"Q\" to exit", (430, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if(len(y_data) > 100):
        y_data = y_data[-100:]
        running_average_array = running_average_array[-100:]

    x = np.linspace(0, len(y_data)-1, len(y_data))
    sigma = 1  # Smoothing parameter
    smooth_y = gaussian_filter1d(y_data, sigma)
    plt.plot(x, y_data, color = 'blue')
    plt.plot(x, running_average_array, color = 'green')
    plt.plot(x, [x*0.9 for x in running_average_array], color = 'red')
    plot_filename = 'plot.png'
    plt.savefig(plot_filename)
    plt.clf()
    image = cv2.imread(plot_filename)
    image = cv2.resize(image, (width, height))
    
    # Combine the images horizontally
    combined_image = cv2.hconcat([frame, image])


    # display the frame
    cv2.imshow("Frame", combined_image)

    # check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
