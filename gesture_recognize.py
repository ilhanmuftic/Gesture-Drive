import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Directory where gesture files are saved
gesture_directory = "gesture_files"

# Load saved gestures
gesture_database = {}
for gesture_file in os.listdir(gesture_directory):
    with open(os.path.join(gesture_directory, gesture_file), 'rb') as file:
        gesture_database[gesture_file] = pickle.load(file)

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        landmarks_list = []
        for landmarks in results.multi_hand_landmarks:
            # Convert landmarks to a NumPy array
            landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark])
            landmarks_list.append(landmarks_np)

            # Draw circles at the locations of all finger tips
            for finger_tip in mp_hands.HandLandmark:
                index = finger_tip.value
                landmark = landmarks.landmark[index]
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Compare with saved gestures
        recognized_gesture = None
        min_mse_value = float('inf')
        for gesture_filename, gesture_landmarks in gesture_database.items():
            mse_value = np.mean((landmarks_list - np.mean(gesture_landmarks, axis=0))**2)
            if mse_value < min_mse_value:
                min_mse_value = mse_value
                recognized_gesture = gesture_filename[0:-4]

        # Display the recognized gesture filename
        cv2.putText(frame, recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Recognizing Gestures', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
