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

        # Draw bounding boxes or circles around detected contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)  # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

                # Optional: Draw a circle around the detected ball
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                center = (int(center_x), int(center_y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 0, 255), 2)  # Draw circle (red)

        # Show the original frame with detections
        cv2.imshow("Orange Ball Detection", frame)

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
