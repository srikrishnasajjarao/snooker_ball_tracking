import cv2
import numpy as np

def main():
    # Path to the video file
    video_path = "videos/sample_video.mp4"
    
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

        # Define the HSV range for detecting blue color
        lower_blue = np.array([100, 150, 50])  # Adjust these values for your video
        upper_blue = np.array([140, 255, 255])

        # Create a mask for blue color
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes or circles around detected contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)  # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle

                # Optional: Draw a circle around the detected ball
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                center = (int(center_x), int(center_y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Show the original frame with detections
        cv2.imshow("Ball Detection", frame)

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
