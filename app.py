import cv2 as cv
import copy
import itertools
import csv
import multiprocessing

from face_verification.verification import verification

from hand_gesture.model.keypoint_classifier import KeyPointClassifier
from hand_gesture.HandTrackingModule import HandDetector, draw_hand, find_distance, calc_bounding_rect
from hand_gesture.control import Speaker, Painter, take_screenshot, open_application

# Initiations ###########################################################
cap_device = 0
cap_height = 900
cap_width = cap_height * (16/9)

# Initiate Subprocess 
screenshot_process = None
application_process = None

# FLAGS ###########################################################
vol_change = False
drawing_mode = False
verified = False
# #################################################################

# Read labels ###########################################################
with open('hand_gesture/model/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]


def main():
    print(keypoint_classifier_labels)
    global screenshot_process, application_process
    global vol_change, drawing_mode, verified

    # Sound Control initiation #############################################
    volume = Speaker()
    
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Model load #############################################################
    mp_model = HandDetector()
    keypoint_classifier = KeyPointClassifier('hand_gesture/model/keypoint_classifier.tflite')

    verifier = verification('face_verification/')
    painter = Painter(height, width, folder_path="hand_gesture/Header")

    #  ####################################################################
    while True:
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        # verification
        if not verified:
            verified = verifier.verify(debug_image)
            if not verified:
                cv.putText(debug_image, "NOT Verified", (int(width * 0.02), int(height * 0.08)), cv.FONT_HERSHEY_COMPLEX,
                       1, (0, 0, 255), 2, cv.LINE_AA)
        else:
            # Detection #############################################################
            right_hand, left_hand = mp_model.find_hands(image)

            if right_hand:
                # classify right hand sign
                landmark_list = right_hand[0:-1]
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                print(hand_sign_id, keypoint_classifier_labels[hand_sign_id])

                if not drawing_mode:    # control mode
                    # draw part
                    draw_hand(debug_image, right_hand, keypoint_classifier_labels[hand_sign_id])
                    # #######################################
                    if hand_sign_id == keypoint_classifier_labels.index('close'):  # take screenshot or quit drawing mode
                        if not screenshot_process or not screenshot_process.is_alive():
                            screenshot_process = multiprocessing.Process(target=take_screenshot)
                            screenshot_process.start()

                    if hand_sign_id == keypoint_classifier_labels.index('select'):    # open text editor
                        if not application_process or not application_process.is_alive():
                            application_process = multiprocessing.Process(target=open_application, args=['text editor', ])
                            application_process.start()

                    if hand_sign_id == keypoint_classifier_labels.index('activate'):    # if "start" sign
                        vol_change = True   # enable volume change

                    if hand_sign_id == keypoint_classifier_labels.index('freeze'):   # if "stop" sign
                        vol_change = False  # disable volume change

                    if hand_sign_id == keypoint_classifier_labels.index('draw'):  # enter drawing mode
                        drawing_mode = True
                    if hand_sign_id == keypoint_classifier_labels.index('lock'):    # go to verification phase
                        verified = False

                else:   # drawing mode
                    x1, y1 = landmark_list[8][:-1]
                    if hand_sign_id == keypoint_classifier_labels.index('select'):  # select colors
                        debug_image = painter.selection(debug_image, x1, y1)

                    if hand_sign_id == keypoint_classifier_labels.index('pen'):  # draw
                        debug_image = painter.drawing(debug_image, x1, y1)

                    if hand_sign_id == keypoint_classifier_labels.index('close'):  # go to control mode
                        drawing_mode = False

            # drawing
            if drawing_mode:
                painter.show_header(debug_image)
                debug_image = painter.paint(debug_image)
            # control sound
            if vol_change and left_hand:
                left_hand_lm = left_hand[0:-1]
                left_hand_border = calc_bounding_rect(image, left_hand[-1])  # right_hand[0]["bbox"]

                distance, debug_image = find_distance(left_hand_lm[4][:-1], left_hand_lm[8][:-1], left_hand_border, debug_image)
                vol = volume(distance)
                cv.putText(debug_image, f"volume: {int(vol)}%", (int(width*0.7), int(height*0.96)), cv.FONT_HERSHEY_COMPLEX,
                           0.8, (255, 255, 0), 1, cv.LINE_AA)
            # ###################################################################################
            # Show text
            if drawing_mode:
                cv.putText(debug_image, "Drawing mode", (int(width*0.02), int(height*0.96)), cv.FONT_HERSHEY_COMPLEX,
                           1, (255, 255, 0), 2, cv.LINE_AA)
            else:
                cv.putText(debug_image, "Control mode", (int(width*0.02), int(height*0.96)), cv.FONT_HERSHEY_COMPLEX,
                           1, (255, 255, 0), 2, cv.LINE_AA)

        # Show window
        # if drawing_mode:
        #     painter.show_windows()
        cv.imshow('Hand Gesture Recognition', debug_image)

        # ###################################################################################


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


if __name__ == '__main__':
    main()
