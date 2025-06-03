import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize x and y
x, y = 0, 0

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    click_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # Convert the image color back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Normalize coordinates
                x = np.interp(index_tip.x, (0, 1), (0, screen_width))
                y = np.interp(index_tip.y, (0, 1), (0, screen_height))

                # Move mouse smoothly
                current_mouse_x, current_mouse_y = pyautogui.position()
                smooth_x = current_mouse_x + 0.3 * (x - current_mouse_x)
                smooth_y = current_mouse_y + 0.3 * (y - current_mouse_y)
                pyautogui.moveTo(smooth_x, smooth_y)

                if thumb_tip.y > index_tip.y:
                    if click_time == 0:
                        click_time = time.time()
                    elif time.time() - click_time > 2:  # if more than 2 seconds have passed
                        pyautogui.click()
                        click_time = 0
                else:
                    click_time = 0

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display coordinates
        cv2.putText(image, f'Normalized coordinates: ({x:.2f}, {y:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the image
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
