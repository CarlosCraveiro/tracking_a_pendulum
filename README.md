# Tracking a Pendulum

This project involves tracking a pendulum’s motion using a **Raspberry Pi Camera V2** and **OpenCV** for image processing. The primary goal is to monitor the pendulum's oscillations and estimate gravitational acceleration by analyzing the average period of the pendulum swing.

![Screenshot](https://github.com/CarlosCraveiro/tracking_a_pendulum/blob/main/images/screenshot.png)

## Project Overview
The **tracking_a_pendulum** project was created as part of an assignment to demonstrate how a simple computer vision setup can be used to extract physical properties from real-world motion. By using image processing techniques to identify the position of a pendulum over time, we calculate the pendulum’s period and use this information to estimate the local gravitational constant, $g$.

This repository contains all code necessary for capturing video, processing images, and calculating the pendulum’s parameters. Additionally, the project setup, configuration details, and underlying physics concepts are documented in the project’s [wiki](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki).

## Project Dependencies
To install the necessary dependencies, run the following commands:
```bash
sudo apt install python3-picamzero
sudo apt install python3-opencv
sudo apt install python3-matplotlib
```

These dependencies include:
- **Picamera2**: For controlling the Raspberry Pi Camera V2 and capturing frames.
- **OpenCV**: For real-time image processing and analysis.
- **Matplotlib**: For plotting data to visualize the pendulum’s motion and analyze oscillations.

## Setup and Equipment
This project is designed to run on a **Raspberry Pi** with the **Raspberry Pi Camera V2**. Here is a basic overview of the setup:

- **Pendulum**: A simple pendulum is used, which is captured and tracked using the camera.
- **Raspberry Pi Camera V2**: Positioned to face the pendulum for real-time tracking.
- **Raspberry Pi**: Processes the camera feed and calculates pendulum parameters using the dependencies mentioned above.

### Setup Images
![Pendulum_Full](https://github.com/CarlosCraveiro/tracking_a_pendulum/blob/main/images/pendulum.jpg)
![RaspiCam](https://github.com/CarlosCraveiro/tracking_a_pendulum/blob/main/images/raspicam.jpg)
![Pendulum_Front](https://github.com/CarlosCraveiro/tracking_a_pendulum/blob/main/images/setup_from_front.jpg)

## Project Wiki
For a deeper understanding of the project, including configuration, physics explanations, and code details, please refer to the [project’s wiki](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki). The wiki contains the following sections:
- **[Camera Setup](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki/Camera-Setup)**: Explanation of the camera configuration and setup.
- **[Code Details](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki/Code-details)**: In-depth breakdown of the code and how each part contributes to the tracking process.
- **[Image Processing](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki/Image-Processing)**: Describes the techniques used for image processing and pendulum tracking.
- **[Pendulum Physics](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki/Pendulum-Physics)**: Covers the physics principles applied to estimate gravitational acceleration from the pendulum’s motion.

## How to Run the Project
1. Set up the pendulum and Raspberry Pi Camera V2 as shown in the images above.
2. Install the dependencies listed in the **Project Dependencies** section.
3. Clone the repository:
   ```bash
   git clone https://github.com/CarlosCraveiro/tracking_a_pendulum.git
   ```
4. Navigate to the project directory and run the main script:
   ```bash
   cd tracking_a_pendulum/src
   python3 main.py
   ```
5. View the output and analyze the pendulum’s motion in real-time through the OpenCV and Matplotlib displays.

This project provides a hands-on approach to learning about pendulum physics and using image processing for motion tracking. Check out the [wiki](https://github.com/CarlosCraveiro/tracking_a_pendulum/wiki) for additional resources and detailed documentation.
