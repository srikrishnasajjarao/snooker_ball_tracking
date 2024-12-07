import cv2
import numpy as np
from collections import deque

def detect_and_draw_ball(frame, mask, points):
    """
    Detect the largest contour in the mask, assume it’s the ball,
    and update the points deque with the ball’s position.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Use the largest contour for better accuracy
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 500:  # Filter small objects
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center = (int(cx), int(cy))
            radius = int(radius)
            
            # Draw circle and bounding box
            cv2.circle(frame, center, radius, (255, 0, 0), 2)
            cv2.rectangle(frame, 
                          (center[0] - radius, center[1] - radius), 
                          (center[0] + radius, center[1] + radius), 
                          (0, 255, 0), 2)
            points.appendleft(center)
            return

    points.appendleft(None)

def draw_path(frame, points):
    """
    Draw the path of the ball using the deque of points.
    """
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        thickness = int(np.sqrt(len(points) / float(i + 1)) * 2.5)
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

def main():
    video_path = "videos/input_video.mp4"
    output_path = "videos/output_video.mp4"

    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # HSV color range for yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Initialize tracking deque
    points = deque(maxlen=512)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and create mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Detect and draw the ball
        detect_and_draw_ball(frame, mask, points)

        # Draw ball path
        draw_path(frame, points)

        # Write frame to output and display
        out.write(frame)
        cv2.imshow("Yellow Ball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to: {output_path}")

if __name__ == "__main__":
    main()
