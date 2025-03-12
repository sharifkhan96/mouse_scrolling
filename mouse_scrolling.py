import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start webcam capture
cap = cv2.VideoCapture(0)
prev_blink_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get right eye landmarks
            right_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            
            # Calculate eye openness
            eye_top = right_eye[1].y * h
            eye_bottom = right_eye[5].y * h
            eye_openness = abs(eye_bottom - eye_top)
            
            # If eye closes for a certain threshold, press "down" for the next slide
            if eye_openness < 3:  
                if time.time() - prev_blink_time > 1:  # Prevent multiple triggers
                    pyautogui.press("down")
                    prev_blink_time = time.time()
    
    cv2.imshow("Face Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
