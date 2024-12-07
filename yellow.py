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

    # Get the original width and height of the video frame
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the new resolution (50% of the original resolution)
    new_width = int(original_width * 0.5)
    new_height = int(original_height * 0.5)

    # Create an OpenCV window and resize it explicitly
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

                # Draw a line to track the ball's movement (only if previous_position exists)
                if previous_position is not None:
                    cv2.line(frame_resized, previous_position, (center_x, center_y), (0, 0, 255), 2)  # Red line

                # Update previous position
                previous_position = (center_x, center_y)

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
