import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load alien face image from a local file
alien_face_path = "C:\\Users\\garde\\Documents\\code\\halloween-video\\green-alien-face-free-png.png"
alien_face = cv2.imread(alien_face_path, cv2.IMREAD_UNCHANGED)

# Check if the alien face image has an alpha channel
if alien_face.shape[2] == 3:
    # Create an alpha channel based on a specific color (e.g., white background)
    alpha_channel = np.ones(alien_face.shape[:2], dtype=alien_face.dtype) * 255
    alien_face = np.dstack((alien_face, alpha_channel))

# Load background image from a local file
background_image_path = "C:\\Users\\garde\\Documents\\code\\halloween-video\\background_image.jpg"
background_image = cv2.imread(background_image_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Alien Background Face", cv2.WINDOW_NORMAL)

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

    # Create a mask for the background
    mask = np.zeros((webcam_height, webcam_width), dtype=np.uint8)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * webcam_width), int(bboxC.ymin * webcam_height), int(bboxC.width * webcam_width), int(bboxC.height * webcam_height)

            # Resize alien face to fit the detected face
            alien_resized = cv2.resize(alien_face, (w, h), interpolation=cv2.INTER_AREA)

            # Create a mask for the alien face
            alien_mask = alien_resized[:, :, 3] / 255.0
            alien_image = alien_resized[:, :, :3]

            # Overlay the alien face on the frame
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (1.0 - alien_mask) * frame[y:y+h, x:x+w, c] + alien_mask * alien_image[:, :, c]

            # Update the mask for the background
            mask[y:y+h, x:x+w] = 255

    # Invert the mask for the background
    mask_inv = cv2.bitwise_not(mask)

    # Extract the background from the frame
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Extract the foreground from the background image
    background_fg = cv2.bitwise_and(background_image, background_image, mask=mask)

    # Combine the frame background and the background foreground
    combined_image = cv2.add(frame_bg, background_fg)

    # Display the resulting frame
    cv2.imshow("Alien Background Face", combined_image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()