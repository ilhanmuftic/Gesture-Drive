import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the camera
cap = cv2.VideoCapture(0)

# Directory to save gesture files
gesture_directory = "gesture_files"
os.makedirs(gesture_directory, exist_ok=True)

# Counter for naming the gesture files
gesture_counter = 1

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

        # Check if a new gesture is significantly different
        is_new_gesture = True
        for gesture_landmarks in os.listdir(gesture_directory):
            # Load existing gestures from files
            with open(os.path.join(gesture_directory, gesture_landmarks), 'rb') as file:
                existing_gesture = pickle.load(file)
            mse_value = np.mean((landmarks_list - np.mean(existing_gesture, axis=0))**2)
            # You may need to adjust the threshold based on your requirements
            if mse_value < 0.01:
                is_new_gesture = False
                break

        if is_new_gesture:
            print(f"New Gesture Captured!")

            # Save the new gesture to a separate file
            gesture_filename = f"gesture_{gesture_counter}.pkl"
            gesture_counter += 1

            with open(os.path.join(gesture_directory, gesture_filename), 'wb') as file:
                pickle.dump(landmarks_list, file)

    # Display the frame
    cv2.imshow('Hand Landmarks', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
