import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.7, tracking_con=0.5):

        self.use_static_image_mode = mode
        self.maxHands = max_hands  
        self.model_complexity = model_complexity
        self.min_detection_confidence = detection_con
        self.min_tracking_confidence = tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.results = None

    def find_hands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # calc landmark list
        if self.results.multi_hand_landmarks:
            all_hands = []  # contains all hands in the image
            for hand_type, hand_landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                my_hand = {"lmList": calc_landmark_list(img, hand_landmarks),
                           "type": hand_type.classification[0].label,
                           "bbox": calc_bounding_rect(img, hand_landmarks)}

                all_hands.append(my_hand)
        else:
            return None
        return all_hands

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv.circle(img, (x1, y1), scale, color, cv.FILLED)
            cv.circle(img, (x2, y2), scale, color, cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv.circle(img, (cx, cy), scale, color, cv.FILLED)

        return length, info, img


def main():
    pTime = 0

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        hands, img = detector.find_hands(img)

        if hands:
            # lmList=hands[0]
            lmList = next((d for d in hands if d.get('type') == 'Right'), None)
            if lmList:
                print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv.imshow("Image", img)
        q = cv.waitKey(1)
        if q == ord("q"):
            break


if __name__ == "__main__":
    main()
