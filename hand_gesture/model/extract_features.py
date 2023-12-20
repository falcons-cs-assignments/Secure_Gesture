import copy
import itertools
import os

import cv2 as cv
import csv
import mediapipe as mp

classes = ['0', '1', '2', '3', '4', '5', '6', 'others']

data_from = "signs_images/augmentation/"

count = {
            '0': len(os.listdir(data_from + "0")),
            '1': len(os.listdir(data_from + "1")),
            '2': len(os.listdir(data_from + "2")),
            '3': len(os.listdir(data_from + "3")),
            '4': len(os.listdir(data_from + "4")),
            '5': len(os.listdir(data_from + "5")),
            '6': len(os.listdir(data_from + "6")),
            'others': len(os.listdir(data_from + "others"))
        }


def logging_csv(number, landmark_list):
    csv_path = './keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])


def main():
    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    for cls in classes:
        folder = data_from + cls + '/'
        num_frames = count[cls]
        sign_id = select_id(cls)
        for img in range(num_frames):
            file = str(img) + '.png'
            frame = cv.imread(os.path.join(folder, file))
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            # ############################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Landmark calculation
                    landmark_list = calc_landmark_list(frame, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    # Write to the dataset file
                    logging_csv(sign_id, pre_processed_landmark_list)


def select_id(cls):
    if cls == 'others':
        return len(classes) - 1
    return int(cls)


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
        if (i + 1) % 3:  # don't normalize the z_coordinate
            return n / max_value
        return n

    temp_landmark_list = list(map(normalize_, range(len(temp_landmark_list)), temp_landmark_list))

    return temp_landmark_list


if __name__ == '__main__':
    main()
