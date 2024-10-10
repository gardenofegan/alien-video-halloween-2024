import cv2
import numpy as np
import yolov5
import sys
import os

# Ensure the utils directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'utils'))

from skeleton_pose_utils import SkeletonDrawer

# Load YOLO model
model = yolov5.load('yolov5s')

# Initialize SkeletonDrawer
skeleton_drawer = SkeletonDrawer()

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLO Skeleton Pose", cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Run YOLO detection
    results = model(frame)

    # Process YOLO results
    for det in results.xyxy[0]:
        if int(det[-1]) == 0:  # Class 0 is person
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Generate dummy keypoints (replace this with actual pose estimation)
            keypoints = np.array([
                [x1 + (x2-x1) * 0.5, y1 + (y2-y1) * 0.1],  # Nose
                [x1 + (x2-x1) * 0.3, y1 + (y2-y1) * 0.2],  # Left eye
                [x1 + (x2-x1) * 0.7, y1 + (y2-y1) * 0.2],  # Right eye
                [x1 + (x2-x1) * 0.4, y1 + (y2-y1) * 0.3],  # Left ear
                [x1 + (x2-x1) * 0.6, y1 + (y2-y1) * 0.3],  # Right ear
                [x1 + (x2-x1) * 0.3, y1 + (y2-y1) * 0.4],  # Left shoulder
                [x1 + (x2-x1) * 0.7, y1 + (y2-y1) * 0.4],  # Right shoulder
                [x1 + (x2-x1) * 0.2, y1 + (y2-y1) * 0.6],  # Left elbow
                [x1 + (x2-x1) * 0.8, y1 + (y2-y1) * 0.6],  # Right elbow
                [x1 + (x2-x1) * 0.1, y1 + (y2-y1) * 0.8],  # Left wrist
                [x1 + (x2-x1) * 0.9, y1 + (y2-y1) * 0.8],  # Right wrist
                [x1 + (x2-x1) * 0.4, y1 + (y2-y1) * 0.7],  # Left hip
                [x1 + (x2-x1) * 0.6, y1 + (y2-y1) * 0.7],  # Right hip
                [x1 + (x2-x1) * 0.4, y1 + (y2-y1) * 0.85],  # Left knee
                [x1 + (x2-x1) * 0.6, y1 + (y2-y1) * 0.85],  # Right knee
                [x1 + (x2-x1) * 0.4, y1 + (y2-y1) * 1.0],  # Left ankle
                [x1 + (x2-x1) * 0.6, y1 + (y2-y1) * 1.0],  # Right ankle
            ], dtype=np.int32)

            # Draw skeleton
            frame = skeleton_drawer.draw_skeleton(frame, keypoints)

    # Display the resulting frame
    cv2.imshow('YOLO Skeleton Pose', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()