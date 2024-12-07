import cv2
import numpy as np

def main():
    # Path to the video file
    video_path = "videos/sample_video.mp4"  # Replace with your video filename
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize variables for ball tracking
    previous_position = None  # To store the previous position of the ball

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Finished playing video.")
            break

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV range for detecting orange color
        lower_orange = np.array([5, 150, 100])  # Lower bound of orange
        upper_orange = np.array([15, 255, 255])  # Upper bound of orange

        # Create a mask for orange color
        mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Track the ball and draw its path
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)  # Bounding box
                center_x, center_y = x + w // 2, y + h // 2  # Center of the bounding box

                # Optional: Draw a circle around the detected ball
                (circle_center_x, circle_center_y), radius = cv2.minEnclosingCircle(contour)
                circle_center = (int(circle_center_x), int(circle_center_y))
                radius = int(radius)
                cv2.circle(frame, circle_center, radius, (0, 0, 255), 2)  # Draw circle (red)

                # Draw a line to track the ball's movement (only if previous_position exists)
                if previous_position is not None:
                    cv2.line(frame, previous_position, (center_x, center_y), (255, 0, 0), 2)

                # Update previous position
                previous_position = (center_x, center_y)

        # Show the original frame with detections and tracking path
        cv2.imshow("Ball Tracking", frame)

        # Show the mask (optional, for debugging)
        cv2.imshow("Mask", mask)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback stopped.")
            break

    # Release video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
