import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the camera
cap = cv2.VideoCapture(0)

# Variables for motion tracking
prev_tip_position = None
sensitivity = 1000  # Adjust this value to control sensitivity
volume = 50  # Initial volume

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
        for landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_position = np.array([index_finger_tip.x, index_finger_tip.y])

            # Draw a circle at the location of the finger tip
            height, width, _ = frame.shape
            cx, cy = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Check motion direction in the X-direction
            if prev_tip_position is not None:
                motion_vector = tip_position - prev_tip_position
                average_x_motion = motion_vector[0]

                # Scale down motion values for less sensitivity
                average_x_motion *= sensitivity

                # Display motion direction
                cv2.putText(frame, f"Average X-Motion: {average_x_motion:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Adjust volume based on motion speed
                if abs(average_x_motion) > 30:
                    # Update volume, limiting it to the range [0, 100]
                    volume = max(0, min(100, volume + np.sign(average_x_motion)))

                    # Display imaginary volume
                    print("Current volume: ", volume)

            # Update previous tip position
            prev_tip_position = tip_position

    # Display the frame
    cv2.imshow('Volume Control Simulation', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
