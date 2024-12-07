import cv2
import numpy as np
from collections import deque

def main():
    # Path to the video file
    video_path = "videos/input_video.mp4"
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize variables for ball tracking
    pts = deque(maxlen=64)  # Deque to store the last 64 ball positions

    # Get the original width and height of the video frame
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the new resolution (50% of the original resolution)
    new_width = int(original_width * 0.5)
    new_height = int(original_height * 0.5)

    # Create OpenCV windows and resize them
    cv2.namedWindow("Yellow Ball Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yellow Ball Tracking", new_width, new_height)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", new_width, new_height)

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Finished playing video.")
            break

        # Resize the frame to the new resolution
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Convert the resized frame to HSV color space
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

        # Define the HSV range for detecting yellow color
        lower_yellow = np.array([20, 150, 100])  # Lower bound of yellow
        upper_yellow = np.array([30, 255, 255])  # Upper bound of yellow

        # Create a mask for yellow color
        mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ball_detected = False  # Flag to check if the ball is detected in this frame

        # Track the ball and draw its path
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)  # Bounding box
                center_x, center_y = x + w // 2, y + h // 2  # Center of the bounding box

                # Optional: Draw a circle around the detected ball
                (circle_center_x, circle_center_y), radius = cv2.minEnclosingCircle(contour)
                circle_center = (int(circle_center_x), int(circle_center_y))
                radius = int(radius)
                cv2.circle(frame_resized, circle_center, radius, (0, 255, 255), 2)  # Draw circle (yellow)

                # Add the ball's current center position to the deque
                pts.appendleft((center_x, center_y))

                ball_detected = True  # Ball detected in this frame

        # If the ball was detected, draw the path
        if len(pts) > 1:
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
                cv2.line(frame_resized, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # Resize the mask to match the frame size
        mask_resized = cv2.resize(mask, (new_width, new_height))

        # Show the resized frames
        cv2.imshow("Yellow Ball Tracking", frame_resized)
        cv2.imshow("Mask", mask_resized)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback stopped.")
            break

    # Release video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
