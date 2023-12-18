import copy
import itertools
import time

import cv2 as cv
import csv
import numpy as np
import mediapipe as mp

cap_device = 0
cap_height = 800
cap_width = cap_height * 1.618
use_brect = True


def logging_csv(number, landmark_list):
    csv_path = './keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])


def main():
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    while True:
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number = select_num(key)
        # ##############################################################
        if number >= 0:
            to_save = 80
            while to_save:
                # Camera capture #####################################################
                ret, image = cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)  # Mirror display
                debug_image = copy.deepcopy(image)
                # Detection implementation #############################################################
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                # ############################################################################
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                          results.multi_handedness):
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        # Write to the dataset file
                        logging_csv(number, pre_processed_landmark_list)
                        to_save -= 1
                        # Drawing part
                        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                        debug_image = draw_landmarks(debug_image, landmark_list)

                    time.sleep(0.5)
                print(f"{80 - to_save} data saved ... the rest is {to_save}")
                cv.imshow('Data Collection', debug_image)

        else:
            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            cv.imshow('Data Collection', image)

    cap.release()
    cv.destroyAllWindows()


def select_num(key):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    return number


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


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(i, n):
        if (i + 1) % 3:    # don't normalize the z_coordinate
            return n / max_value
        return n

    temp_landmark_list = list(map(normalize_, range(len(temp_landmark_list)), temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2][:-1]), tuple(landmark_point[3][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2][:-1]), tuple(landmark_point[3][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3][:-1]), tuple(landmark_point[4][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3][:-1]), tuple(landmark_point[4][:-1]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5][:-1]), tuple(landmark_point[6][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5][:-1]), tuple(landmark_point[6][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6][:-1]), tuple(landmark_point[7][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6][:-1]), tuple(landmark_point[7][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7][:-1]), tuple(landmark_point[8][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7][:-1]), tuple(landmark_point[8][:-1]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9][:-1]), tuple(landmark_point[10][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9][:-1]), tuple(landmark_point[10][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10][:-1]), tuple(landmark_point[11][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10][:-1]), tuple(landmark_point[11][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11][:-1]), tuple(landmark_point[12][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11][:-1]), tuple(landmark_point[12][:-1]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13][:-1]), tuple(landmark_point[14][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13][:-1]), tuple(landmark_point[14][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14][:-1]), tuple(landmark_point[15][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14][:-1]), tuple(landmark_point[15][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15][:-1]), tuple(landmark_point[16][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15][:-1]), tuple(landmark_point[16][:-1]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17][:-1]), tuple(landmark_point[18][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17][:-1]), tuple(landmark_point[18][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18][:-1]), tuple(landmark_point[19][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18][:-1]), tuple(landmark_point[19][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19][:-1]), tuple(landmark_point[20][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19][:-1]), tuple(landmark_point[20][:-1]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0][:-1]), tuple(landmark_point[1][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0][:-1]), tuple(landmark_point[1][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1][:-1]), tuple(landmark_point[2][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1][:-1]), tuple(landmark_point[2][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2][:-1]), tuple(landmark_point[5][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2][:-1]), tuple(landmark_point[5][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5][:-1]), tuple(landmark_point[9][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5][:-1]), tuple(landmark_point[9][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9][:-1]), tuple(landmark_point[13][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9][:-1]), tuple(landmark_point[13][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13][:-1]), tuple(landmark_point[17][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13][:-1]), tuple(landmark_point[17][:-1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17][:-1]), tuple(landmark_point[0][:-1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17][:-1]), tuple(landmark_point[0][:-1]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index in [4, 8, 12, 16, 20]:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        else:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    return image


if __name__ == '__main__':
    main()
