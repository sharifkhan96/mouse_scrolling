import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start webcam capture
cap = cv2.VideoCapture(0)
prev_left_blink_time = time.time()
prev_right_blink_time = time.time()

# Define eye landmarks
# For MediaPipe, right eye landmarks (anatomical right, left side of image)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Left eye landmarks (anatomical left, right side of image)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Status text
status_text = "Ready"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get left and right eye landmarks
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
            
            # Calculate eye openness for both eyes
            right_eye_top = right_eye[1].y * h
            right_eye_bottom = right_eye[5].y * h
            right_eye_openness = abs(right_eye_bottom - right_eye_top)
            
            left_eye_top = left_eye[1].y * h
            left_eye_bottom = left_eye[5].y * h
            left_eye_openness = abs(left_eye_bottom - left_eye_top)
            
            # Display eye openness values
            cv2.putText(frame, f"Right eye: {right_eye_openness:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Left eye: {left_eye_openness:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Right eye blink (previous slide)
            if right_eye_openness < 3:
                if time.time() - prev_right_blink_time > 1:  # Prevent multiple triggers
                    pyautogui.press("up")
                    status_text = "Previous slide"
                    prev_right_blink_time = time.time()
            
            # Left eye blink (next slide)
            if left_eye_openness < 3:
                if time.time() - prev_left_blink_time > 1:  # Prevent multiple triggers
                    pyautogui.press("down")
                    status_text = "Next slide"
                    prev_left_blink_time = time.time()
    
    # Display status
    cv2.putText(frame, f"Status: {status_text}", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Eye-Controlled Presentation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
