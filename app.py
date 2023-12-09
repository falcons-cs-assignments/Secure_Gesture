import cv2 as cv
import copy
import itertools
import csv

# import numpy as np

from model.keypoint_classifier import KeyPointClassifier
from HandTrackingModule import HandDetector, draw_hand, find_distance
from control import Speaker

cap_device = 0
cap_height = 900
cap_width = cap_height * 1.618

vol_change = False

# Read labels ###########################################################
with open('model/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]


def main():
    global vol_change
    # Sound Control initiation #############################################
    volume = Speaker()
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_model = HandDetector()
    keypoint_classifier = KeyPointClassifier()

    #  ####################################################################
    while True:
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        # Detection #############################################################
        hands = mp_model.find_hands(image)
        if hands:
            # split right and left hands
            right_hand = [h for h in hands if h["type"] == "Right"]
            left_hand = [h for h in hands if h["type"] == "Left"]

            # #############################
            # classify left hand sign
            if len(left_hand):
                landmark_list = left_hand[0]['lmList']
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                print(hand_sign_id, keypoint_classifier_labels[hand_sign_id])

                # draw part
                draw_hand(debug_image, left_hand[0], keypoint_classifier_labels[hand_sign_id])
                # #######################################
                # control part
                # TODO: switching depend on left hand_sign_id
                if hand_sign_id == 0:    # if "start" sign
                    vol_change = True   # enable volume change
                if hand_sign_id == 1:   # if "stop" sign
                    vol_change = False  # disable volume change

            # control sound
            if vol_change and len(right_hand):
                right_hand_lm = right_hand[0]["lmList"]
                right_hand_border = right_hand[0]["bbox"]

                distance, debug_image = find_distance(right_hand_lm[4], right_hand_lm[8], right_hand_border, debug_image)
                vol = volume(distance)
                print(round(distance, 4), ":", vol)
            # ##############################################

        cv.imshow('Hand Gesture Recognition', debug_image)

        #  ####################################################################


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


if __name__ == '__main__':
    main()
