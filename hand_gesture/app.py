import cv2 as cv
import copy
import itertools
import csv
import multiprocessing

from model.keypoint_classifier import KeyPointClassifier
from HandTrackingModule import HandDetector, draw_hand, find_distance, calc_bounding_rect
from control import Speaker, Painter, take_screenshot, open_application

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
selection = False
pen = False
termination = False
quit = True

hand_flag = False
# #################################################################

# Read labels ###########################################################
with open('model/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]


def main():
    global screenshot_process, application_process
    global vol_change, drawing_mode, selection, pen, quit, termination, hand_flag

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
    keypoint_classifier = KeyPointClassifier()

    painter = Painter(height, width)

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
        right_hand, left_hand = mp_model.find_hands(image)
        
        if left_hand:

            # classify left hand sign
            landmark_list = left_hand[0:-1]
            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            print(hand_sign_id, keypoint_classifier_labels[hand_sign_id])

            if not drawing_mode:
                # draw part
                draw_hand(debug_image, left_hand, keypoint_classifier_labels[hand_sign_id])
            # #######################################
            # control part
            # TODO: switching depend on left hand_sign_id
            if hand_sign_id == keypoint_classifier_labels.index('quit'):  # take screenshot or quit drawing mode
                if not drawing_mode:
                    if not screenshot_process or not screenshot_process.is_alive():
                        screenshot_process = multiprocessing.Process(target=take_screenshot)
                        screenshot_process.start()
                
                else:
                    drawing_mode = False

            if hand_sign_id == keypoint_classifier_labels.index('open_app'):    # open text editor
                if not drawing_mode:
                    # open_application("text editor")
                    if not application_process or not application_process.is_alive():
                        application_process = multiprocessing.Process(target=open_application, args=['text editor', ])
                        application_process.start()

                pen = False
                selection = False
                hand_flag = True

            if hand_sign_id == keypoint_classifier_labels.index('start'):    # if "start" sign
                if not drawing_mode:
                    vol_change = True   # enable volume change

                pen = False
                selection = False
                hand_flag = True
           
            if hand_sign_id == keypoint_classifier_labels.index('lock'):   # if "stop" sign
                if not drawing_mode:
                    vol_change = False  # disable volume change
                
                pen = False
                selection = False
                hand_flag = True

            if hand_sign_id == keypoint_classifier_labels.index('draw'):  # enter drawing mode
                drawing_mode = True
                pen = False
                selection = False
                hand_flag = True

            if hand_sign_id == keypoint_classifier_labels.index('select'):  # select colors
                selection = True
                pen = False

            if hand_sign_id == keypoint_classifier_labels.index('pen'):  # draw
                pen = True
                selection = False

            if hand_sign_id == keypoint_classifier_labels.index('terminate'):  # close program
                # if drawing_mode:
                #     drawing_mode = False
                # else:
                termination = True
                pen = False
                selection = False

        # control sound
        if vol_change and right_hand:
            right_hand_lm = right_hand[0:-1]
            right_hand_border = calc_bounding_rect(image, right_hand[-1])  # right_hand[0]["bbox"]

            distance, debug_image = find_distance(right_hand_lm[4][:-1], right_hand_lm[8][:-1], right_hand_border, debug_image)
            vol = volume(distance)
            print(round(distance, 4), ":", vol)
            cv.putText(debug_image, f"volume: {int(vol)}%", (int(width*0.7), int(height*0.96)), cv.FONT_HERSHEY_COMPLEX,
                       0.8, (255, 255, 0), 1, cv.LINE_AA)
        # ###################################################################################
        # drawing mode
        if drawing_mode:
            painter.show_header(debug_image)
            if right_hand or left_hand:
                lmList = right_hand[0:-1] or left_hand[0:-1]
                x1, y1 = lmList[8][:-1]

                if selection:
                    debug_image = painter.selection(debug_image, x1, y1)
                
                if pen:
                    debug_image = painter.drawing(debug_image, x1, y1, hand_flag)
                    hand_flag = False
            else:
                hand_flag = True
            
            debug_image = painter.paint(debug_image)
        # ###################################################################################
            
        # Display ###########################################################################
            
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
