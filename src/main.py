import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from time import time
import threading
import queue
import signal
import sys

# Initialize queues for communication between threads
frame_queue = queue.Queue(maxsize=10)
processed_queue = queue.Queue(maxsize=10)

# Initialize Picamera2 and configure the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# OpenCV trackbar parameters
max_value = 100
max_binary_value = 255
max_threshold_type = 4
max_threshold_value = 255
max_elem = 2
max_kernel_size = 21
max_ampl_thrs = 100
window_name = "Color, Threshold, Morphology Filter"

# Standard Values
red_filter = 255
green_filter = 0
blue_filter = 0
threshold_type = 0
threshold_value = 80
kernel_shape = 2
kernel_size = 4
default_ampl_thrs = 10

# Create OpenCV window
cv2.namedWindow(window_name)

# Create trackbars for color filtering
cv2.createTrackbar("Blue", window_name, blue_filter, max_value, lambda x: None)
cv2.createTrackbar("Green", window_name, green_filter, max_value, lambda x: None)
cv2.createTrackbar("Red", window_name, red_filter, max_value, lambda x: None)

# Create trackbars for thresholding
cv2.createTrackbar('Threshold Type', window_name, threshold_type, max_threshold_type, lambda x: None)
cv2.createTrackbar('Threshold Value', window_name, threshold_value, max_threshold_value, lambda x: None)

# Create trackbars for morphology operations
cv2.createTrackbar('Element Shape', window_name, kernel_shape, max_elem, lambda x: None)
cv2.createTrackbar('Kernel Size', window_name, kernel_size, max_kernel_size, lambda x: None)

# Create trackbars for amplitude threshold for computing period
cv2.createTrackbar('Amplitude percent', window_name, default_ampl_thrs, max_ampl_thrs, lambda x: None)

# Constants for gravity calculation
L = 0.7  # Length of the pendulum in meters

# Some parameters
timev = []
pos = []
init_time = time()
periods = []

# Morphological shape mapper
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

# Function for capturing frames
def capture_frames():
    while True:
        frame = picam2.capture_array()
        if frame_queue.full():
            continue
        frame_time = time() - init_time
        frame_queue.put((frame, frame_time))

# Function for processing frames
def process_frames():
    while True:
        frame, frame_time = frame_queue.get()
        if frame is None:
            break

        # Get trackbar positions
        blue_filter = cv2.getTrackbarPos("Blue", window_name) / max_value
        green_filter = cv2.getTrackbarPos("Green", window_name) / max_value
        red_filter = cv2.getTrackbarPos("Red", window_name) / max_value
        threshold_type_value = cv2.getTrackbarPos('Threshold Type', window_name)
        threshold_value_setting = cv2.getTrackbarPos('Threshold Value', window_name)
        morph_shape_val = morph_shape(cv2.getTrackbarPos('Element Shape', window_name))
        kernel_val = cv2.getTrackbarPos('Kernel Size', window_name)

        # Apply the color filters by scaling each channel
        filtered_im = frame.copy()
        filtered_im[:, :, 0] = np.clip(filtered_im[:, :, 0] * blue_filter, 0, 255)
        filtered_im[:, :, 1] = np.clip(filtered_im[:, :, 1] * green_filter, 0, 255)
        filtered_im[:, :, 2] = np.clip(filtered_im[:, :, 2] * red_filter, 0, 255)

        # Convert to grayscale and apply threshold
        gray_im = cv2.cvtColor(filtered_im, cv2.COLOR_BGR2GRAY)
        _, thresholded_im = cv2.threshold(gray_im, threshold_value_setting, max_binary_value, threshold_type_value)

        # Apply morphological transformations
        element = cv2.getStructuringElement(morph_shape_val, (2 * kernel_val + 1, 2 * kernel_val + 1), (kernel_val, kernel_val))
        eroded_im = cv2.erode(thresholded_im, element)
        dilated_im = cv2.dilate(eroded_im, element)

        # Calculate centroid
        try:
            M = cv2.moments(dilated_im)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(dilated_im, (cX, cY), 5, (0, 0, 255), -1)
        except ZeroDivisionError:
            cX, cY = -1, -1

        # Send processed frame and centroid position to the display thread
        if processed_queue.full():
            continue 
        processed_queue.put((dilated_im, cX, frame_time))

# Signal handler for clean exit
def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    picam2.stop()
    cv2.destroyAllWindows()
    plt.close('all')
    sys.exit(0)

# Set up signal handling for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize and start capture and processing threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)
capture_thread.start()
process_thread.start()

# Main thread handles matplotlib plotting and display
plt.ion()

while True:
    try:
        # Retrieve processed frame and centroid data
        processed_frame,  cX, current_time = processed_queue.get(timeout=1)
        
        # Plotting the centroid positions over time
        if cX >= 0:
            pos.append(cX)
            #current_time = time() - init_time
            timev.append(current_time)
            ampl_perct = cv2.getTrackbarPos('Amplitude percent', window_name) / max_ampl_thrs
            ampl_abs = 0
            # Detecting oscillations by checking time intervals between two maximuns
            if len(pos) > 10:
                max_pos = pos[len(pos) - 1]
                min_pos = max_pos
                amplitude = 0
                for i in range(1, 10):
                    # find the amplitude of the oscilation
                    if(max_pos < pos[len(pos) - i]):
                        max_pos = pos[len(pos) - i]
                    if(min_pos > pos[len(pos) - i]):
                        min_pos = pos[len(pos) - i]
                amplitude = (max_pos - min_pos)
                ampl_abs = max_pos - ampl_perct*amplitude
                # Computing oscilating period 
                found_max = 0
                found_min = 0
                first_found_index = 0
                for i in range(1, 10):
                    if (found_min == 1) & (found_max == 1) & ((max_pos - pos[len(pos) - i]) < ampl_perct*amplitude):
                        period = timev[len(timev) - first_found_index] - timev[len(timev) - i]
                        print(f"Period: {period:.2f} s")
                        print(f"Amplitude: {amplitude:.2f} pixels")
                        periods.append(period)
                        if len(periods) > 10:
                            periods.pop(0)
                        break
                    if((found_max == 1) & ((pos[len(pos) - i] - min_pos) < ampl_perct*amplitude)):
                        found_min = 1
                    if((max_pos - pos[len(pos) - i]) < ampl_perct*amplitude):
                        found_max = 1
                        first_found_index = i
 
            # Calculating g based on last 10 oscillations
            if len(periods) >= 10:
                avg_period = sum(periods) / len(periods)
                g = 4 * np.pi**2 * L / avg_period**2
                print(f"Estimated gravity: {g:.4f} m/s²")
                delta_t = timev[len(timev) - 1] - timev[len(timev) -2]
                fps = 1/delta_t # Assuming sampling rate constant
                print(f"Approx FPS: {fps:.1f}")

            # Plot
            plt.clf()
            plt.plot(timev, pos, color='b')
            plt.axhline(y=(ampl_abs), color='r', linestyle='--', label='Amplitude Threshold')  # Add horizontal line
            plt.title("Centroid X position over time")
            plt.xlabel("Time (s)")
            plt.ylabel("X Position")
            plt.draw()
            plt.pause(0.001)

        # Display processed image using OpenCV
        cv2.imshow(window_name, processed_frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except queue.Empty:
        continue

# Clean up
picam2.stop()
cv2.destroyAllWindows()
plt.show(block=True)

