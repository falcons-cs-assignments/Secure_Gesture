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
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

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


def draw_bounding_rect(image, landmark_point, text=''):

    brect = calc_bounding_rect(image, landmark_point)

    cv.rectangle(image, (brect[0] - 20, brect[1] - 20),
                 (brect[2] + 20, brect[3] + 20), (0, 0, 0), 1)

    if text:
        cv.rectangle(image, (brect[0] - 20, brect[1] - 20), (brect[2] + 20, brect[1] - 44),
                     (0, 0, 0), -1)
        cv.putText(image, text, (brect[0] - 20, brect[1] - 27), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_landmarks(image, landmark_point):
    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0))
    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0))
    
    mp.solutions.drawing_utils.draw_landmarks(image, landmark_point,
                                              mp.solutions.hands.HAND_CONNECTIONS,
                                              landmark_drawing_spec=landmark_drawing_spec,
                                              connection_drawing_spec=connection_drawing_spec
                                              )
    return image


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

        self.results = None

    def find_hands(self, img, draw=True):
        """
        :param img:
        :param draw:
        :return: rightHand [[],[],.....[], handLms]
                to get landmarks rightHand[0,-1]
                to get normalized landmarks rightHand[-1]
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        rightHand = []
        leftHand = []
        # calc landmark list
        if self.results.multi_hand_landmarks:
            for hand_type, hand_landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                tempList = calc_landmark_list(img, hand_landmarks)
                tempList.append(hand_landmarks)

                if hand_type.classification[0].label == "Right":
                    rightHand = tempList
                else:
                    leftHand = tempList
        #
        return rightHand, leftHand


def find_distance(p1, p2, hand_border, img=None, color=(255, 0, 255), scale=5):
    """
    :p1 & p2: should be (x, y)
    """
    # TODO: solve the problem of distance from the camera
    border_rad = math.hypot(hand_border[0] - hand_border[2], hand_border[1] - hand_border[3])

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    ratio = length / border_rad
    if img is not None:
        cv.circle(img, (x1, y1), scale, color, cv.FILLED)
        cv.circle(img, (x2, y2), scale, color, cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
        cv.circle(img, (cx, cy), scale, color, cv.FILLED)

    return ratio, img


def draw_hand(img, hand, label):
    img = draw_bounding_rect(img, hand[-1], label)
    img = draw_landmarks(img, hand[-1])
    return img


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        
        cv.imshow("Image", img)
        q = cv.waitKey(1)
        if q == ord("q"):
            break


if __name__ == "__main__":
    main()
