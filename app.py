import cv2

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

        # Display the frame
        cv2.imshow("Video", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback stopped.")
            break

    # Release video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
