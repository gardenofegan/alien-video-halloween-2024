# References: 
# - Original model: https://google.github.io/mediapipe/solutions/face_detection.html
# - Alien Face image: https://static.vecteezy.com/system/resources/previews/026/958/196/non_2x/green-alien-face-free-png.png

import cv2
import mediapipe as mp
import numpy as np
from imread_from_url import imread_from_url

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load alien face image
alien_face_url = "C:\\Users\\garde\\Documents\\code\\halloween-video\\green-alien-face-free-png.png"
alien_face = cv2.imread(alien_face_url, cv2.IMREAD_UNCHANGED)

# Check if the alien face image has an alpha channel
if alien_face.shape[2] == 3:
    # Create an alpha channel based on a specific color (e.g., white background)
    alpha_channel = np.ones(alien_face.shape[:2], dtype=alien_face.dtype) * 255
    alien_face = np.dstack((alien_face, alpha_channel))

# Load background image
background_image_url = "https://friendlystock.com/wp-content/uploads/2019/07/1-barren-alien-world-space-background-cartoon-clipart.jpg"
background_image = imread_from_url(background_image_url)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Alien Face with Background", cv2.WINDOW_NORMAL)

# Resize the background image to the resolution of the webcam
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
background_image = cv2.resize(background_image, (webcam_width, webcam_height), interpolation=cv2.INTER_AREA)

while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face detection
    results = face_detection.process(rgb_frame)

    # Start with the background image
    combined_image = background_image.copy()

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * webcam_width), int(bboxC.ymin * webcam_height), int(bboxC.width * webcam_width), int(bboxC.height * webcam_height)

            # Resize alien face to fit the detected face
            alien_resized = cv2.resize(alien_face, (w, h), interpolation=cv2.INTER_AREA)

            # Create a mask for the alien face
            alien_mask = alien_resized[:, :, 3] / 255.0
            alien_image = alien_resized[:, :, :3]

            # Overlay the alien face on the combined image using the alpha mask
            for c in range(3):
                combined_image[y:y+h, x:x+w, c] = (1.0 - alien_mask) * combined_image[y:y+h, x:x+w, c] + alien_mask * alien_image[:, :, c]

    # Display the resulting frame
    cv2.imshow("Alien Face with Background", combined_image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()

