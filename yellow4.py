import cv2
import numpy as np
from collections import deque

def main():
    # Path to the input video file
    video_path = "videos/input_video.mp4"
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize a deque to store the last 64 positions of the ball
    pts = deque(maxlen=512)

    # Retrieve video properties: width, height, and frames per second (fps)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define new resolution (50% of the original)
    new_width, new_height = original_width // 2, original_height // 2

    # Set up video writer for saving output
    output_path = "videos/output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Loop through frames of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished playing video.")
            break

        # Resize the frame to the new resolution
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Convert the frame to HSV color space for color detection
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting yellow color
        lower_yellow = np.array([20, 150, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Find contours of the detected yellow regions
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour to detect the ball
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                # Get the center and radius of the enclosing circle
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                # Draw a circle around the detected ball
                cv2.circle(frame_resized, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
                
                # Add the center of the ball to the deque
                pts.appendleft((int(cx), int(cy)))

        # Draw the path of the ball using the deque
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            # Set line thickness based on the age of the point
            thickness = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
            cv2.line(frame_resized, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # Write the annotated frame to the output video
        out.write(frame_resized)

        # Display the frame with tracking annotations
        cv2.imshow("Yellow Ball Tracking", frame_resized)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at: {output_path}")

if __name__ == "__main__":
    main()
