import math
import time
import cv2
import mediapipe as mp
import autopy as ap
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

radius = 8

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 8


upper_left = (0, 0)
bottom_left = (0, 0)


def isUp(y1, y2):
    try:
        dist = y2 - y1
        if y2 > y1:
            return True
        return False
    except Exception:
        print("Getting exception")
        return False


def drawCircleIfUp(image, x1, y1, x2, y2):
    fingerIsUp = isUp(y1 * screen_height, y2 * screen_height)
    if fingerIsUp:
        center_coordinates = (int(x1 * image_width), int(y1 * image_height))
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image


def drawLine(image, x1, y1, x2, y2):
    image = cv2.line(image, (int(x1 * image_width), int(y1 * image_height)), (int(
        x2 * image_width), int(y2 * image_height)), (255, 0, 0), thickness=2, lineType=8)
    return image


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)) * 100


def getFingersTipPos(landmarks):
    indexFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    indexFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    midFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    midFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ringFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ringFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinkyFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinkyFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    f1 = isUp(indexFingerTipLandmarks.y * screen_height,
              indexFingerDipLandmarks.y * screen_height)
    f2 = isUp(midFingerTipLandmarks.y * screen_height,
              midFingerDipLandmarks.y * screen_height)
    f3 = isUp(ringFingerTipLandmarks.y * screen_height,
              ringFingerDipLandmarks.y * screen_height)
    f4 = isUp(pinkyFingerTipLandmarks.y * screen_height,
              pinkyFingerDipLandmarks.y * screen_height)
    return [f1, f2, f3, f4]


def moveMouse(x1, y1):
    try:
        moveX = int(x1 * image_width)
        moveY = int(x1 * image_height)
        #     x1 = int(x1 * screen_width)
        #     y1 = int(y1 * screen_height)
        x1 = np.interp(moveX, [image_width * 0.10,
                               image_width * 0.90], [0, screen_width])
        y1 = np.interp(moveX, [image_height * 0.10,
                               image_height * 0.90], [0, screen_height])
        ap.mouse.smooth_move(x1, y1)
    except Exception:
        print(x1, y1)


def performClickOperation():
    pass
#     ap.mouse.click()


screen_width, screen_height = ap.screen.size()
# screen_width = pai.size().width
# screen_height = pai.size().height
landmarks = 0
pTime = 0
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        max_num_hands=1,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, channels = image.shape

        upper_left = (int(image_height * 0.10), int(image_width * 0.10))
        bottom_right = (int(image_width * 0.90), int(image_height * 0.90))
        # draw in the image
        image = cv2.rectangle(
            image, upper_left, bottom_right, (0, 255, 0), thickness=1)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks
                indexFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                indexFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                midFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                midFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                ringFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ringFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                pinkyFingerTipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinkyFingerDipLandmarks = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
                image = drawCircleIfUp(image, indexFingerTipLandmarks.x, indexFingerTipLandmarks.y,
                                       indexFingerDipLandmarks.x, indexFingerDipLandmarks.y)
                image = drawCircleIfUp(image, midFingerTipLandmarks.x, midFingerTipLandmarks.y,
                                       midFingerDipLandmarks.x, midFingerDipLandmarks.y)
                image = drawCircleIfUp(image, ringFingerTipLandmarks.x, ringFingerTipLandmarks.y,
                                       ringFingerDipLandmarks.x, ringFingerDipLandmarks.y)
                image = drawCircleIfUp(image, pinkyFingerTipLandmarks.x, pinkyFingerTipLandmarks.y,
                                       pinkyFingerDipLandmarks.x, pinkyFingerDipLandmarks.y)

                twoFingDistance = euclideanDistance(
                    indexFingerTipLandmarks.x, indexFingerTipLandmarks.y, midFingerTipLandmarks.x, midFingerTipLandmarks.y)
                fingers = getFingersTipPos(hand_landmarks.landmark)
                print(twoFingDistance)
                if fingers[0] and fingers[1] and twoFingDistance < 5:
                    print("Click Operation")
                    image = drawLine(image, indexFingerTipLandmarks.x, indexFingerTipLandmarks.y,
                                     midFingerTipLandmarks.x, midFingerTipLandmarks.y)
                    performClickOperation()
                elif fingers[0] and fingers[1]:
                    image = drawLine(image, indexFingerTipLandmarks.x, indexFingerTipLandmarks.y,
                                     midFingerTipLandmarks.x, midFingerTipLandmarks.y)
                elif fingers[0]:
                    moveMouse(indexFingerTipLandmarks.x,
                              indexFingerTipLandmarks.y)
                    print("Mouse work")
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
