import cv2
import numpy as np
import pyautogui
import autopy
import HandTrackingV2 as htm
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time  # For controlling smooth scrolling intervals

# Screen dimensions and settings
wCam, hCam = 640, 480
frameR = 100  # Frame reduction for interaction space
smoothening = 7

# Initialize timing and coordinate variables
pTime = 0
plocX, plocY, clocX, clocY = 0, 0, 0, 0

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector and screen size
detector = htm.handDetector(maxHands=2)
wScr, hScr = autopy.screen.size()

# Initialize zooming variables
zooming_mode = False
initial_distance = 0

# Initialize Mediapipe hands object only once
mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)

# Adjust scroll speed factor
scroll_speed = 500  # You can increase this value for faster scrolling
scroll_steps = 2  # Number of steps to break the scrolling into (more steps = smoother scroll)
scroll_interval = 0.001  # Time between each step for smooth scrolling (in seconds)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip image for mirror effect

    # Convert BGR image to RGB for Mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(imgRGB)

    # Hand zooming and scrolling functionality
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        zooming_mode = True
        hand1_lmList, hand2_lmList = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]

        # Convert landmarks to pixel coordinates
        hand1_pos = [(int(lm.x * wCam), int(lm.y * hCam)) for lm in hand1_lmList.landmark]
        hand2_pos = [(int(lm.x * wCam), int(lm.y * hCam)) for lm in hand2_lmList.landmark]

        # Get index finger coordinates for both hands (landmark 8)
        hand1_index_finger = hand1_pos[8]
        hand2_index_finger = hand2_pos[8]

        # Visualize the index fingers
        cv2.circle(img, hand1_index_finger, 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, hand2_index_finger, 15, (255, 0, 0), cv2.FILLED)

        # Calculate the distance between index fingers
        current_distance = np.linalg.norm(np.array(hand1_index_finger) - np.array(hand2_index_finger))

        # Set initial distance for zooming
        if initial_distance == 0:
            initial_distance = current_distance

        # Compare distances to determine zoom in or out
        zoom_change = current_distance - initial_distance
        if zoom_change > 50:  # Zoom in threshold
            pyautogui.hotkey('ctrl', '+')
            initial_distance = current_distance
            cv2.putText(img, "Zooming In", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        elif zoom_change < -50:  # Zoom out threshold
            pyautogui.hotkey('ctrl', '-')
            initial_distance = current_distance
            cv2.putText(img, "Zooming Out", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
        zooming_mode = False
        initial_distance = 0  # Reset zooming state if hands are not detected

    # Handle single-hand detection for scrolling or clicking
    if results.multi_handedness and len(results.multi_handedness) == 1:
        hand_label = MessageToDict(results.multi_handedness[0])['classification'][0]['label']

        # Left hand for scrolling
        if hand_label == 'Left':
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if lmList:
                try:
                    # Extract finger coordinates (index: 8, middle: 12)
                    x1, y1 = lmList[0][8][1:3]  # Index finger
                    x2, y2 = lmList[0][12][1:3]  # Middle finger

                    # Check finger status
                    fingers = detector.fingersUp()

                    # Scroll if only index finger is up
                    if fingers[1] == 1 and fingers[2] == 0:
                        if y1 < hCam // 2:  # Scroll up
                            for _ in range(scroll_steps):  # Smooth scroll loop
                                pyautogui.scroll(scroll_speed // scroll_steps)  # Smaller increments
                                time.sleep(scroll_interval)  # Pause for smoothness
                            cv2.putText(img, "Scrolling Up", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        elif y1 > hCam // 2:  # Scroll down
                            for _ in range(scroll_steps):  # Smooth scroll loop
                                pyautogui.scroll(-scroll_speed // scroll_steps)  # Smaller increments
                                time.sleep(scroll_interval)  # Pause for smoothness
                            cv2.putText(img, "Scrolling Down", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    # Check for click action when both fingers are up
                    if fingers[1] == 1 and fingers[2] == 1:
                        length, img, lineInfo = detector.findDistance(8, 12, img)
                        if length < 40:  # Simulate a mouse click
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            autopy.mouse.click()
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")
        else:
            # No actions for the right hand
            cv2.putText(img, "Right Hand Detected: No Action", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display the processed image
    cv2.imshow("Hand Tracking Interface", img)

    # Exit the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
