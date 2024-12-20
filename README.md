# snooker_ball_tracking

This project tracks a yellow snooker ball in a video using computer vision techniques. The program processes the input video, identifies the yellow ball, and draws a bounding circle, a bounding box, and a path representing the ball's movement. The processed video is saved as an output file.

## Features
1. Detects and tracks the movement of a yellow snooker ball.
2. Draws a bounding circle and path to represent the ball's trajectory.
3. Processes input videos and outputs the result with overlays.

## Folder Structure

![Folder Structure](venv/carbon.png)

## Prerequisites
1. Python 3.9 or higher installed on your system.
2. Virtual environment installed (optional but recommended).

## Installation

1. Clone this repository to your local machine
![Folder Structure](venv/step1.png)  
2. Set up a virtual environment (recommended)
![Folder Structure](venv/step2.png)  
3. Install dependencies using pip
![Folder Structure](venv/step3.png)  

## Usage
1. ##### Place the input video:
Ensure the input video (input_video.mp4) is placed in the videos/ folder.

2. ##### Run the program:
Execute the main script to process the video.

3. ##### Output video:
The processed video with tracking overlays will be saved as output_video.mp4 in the videos/ folder.

4. ##### View the output:
The program will display the live tracking progress in a window. Press q to quit early.

## How It Works
1. ##### HSV Masking:
   The program converts the video frames to HSV color space and creates a mask for detecting yellow objects.
   
2. ##### Contour Detection:
   It identifies the largest contour matching the yellow color and assumes it to be the ball.
   
3. ##### Visualization:
   The program draws a bounding circle and a rectangle around the ball and overlays the ball's trajectory path on the frames.
   
4. ##### Output:
   The annotated frames are saved to an output video file.
