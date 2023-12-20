import os
import re
import cv2 as cv


def main():
    cap = cv.VideoCapture(0)
    directory = 'Image/'
    left = True
    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        # key stroke ####################
        interrupt = cv.waitKey(10)
        if interrupt == 27:  # ESC
            break
        if interrupt & 0xFF == ord('l'):
            left = True
        if interrupt & 0xFF == ord('r'):
            left = False
        # ###############################
        count = {
            '0': {'r': len([f for f in os.listdir(directory + "0") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "0") if f.startswith("L_")])},
            '1': {'r': len([f for f in os.listdir(directory + "1") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "1") if f.startswith("L_")])},
            '2': {'r': len([f for f in os.listdir(directory + "2") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "2") if f.startswith("L_")])},
            '3': {'r': len([f for f in os.listdir(directory + "3") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "3") if f.startswith("L_")])},
            '4': {'r': len([f for f in os.listdir(directory + "4") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "4") if f.startswith("L_")])},
            '5': {'r': len([f for f in os.listdir(directory + "5") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "5") if f.startswith("L_")])},
            '6': {'r': len([f for f in os.listdir(directory + "6") if f.startswith("R_")]),
                  'l': len([f for f in os.listdir(directory + "6") if f.startswith("L_")])},
            'others': {'r': len([f for f in os.listdir(directory + "others") if f.startswith("R_")]),
                       'l': len([f for f in os.listdir(directory + "others") if f.startswith("L_")])}
        }

        if left:
            cv.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
            cv.imshow("data", frame)
            cv.imshow("ROI", frame[40:400, 0:300])
            frame = frame[40:400, 0:300]
        else:
            cv.rectangle(frame, (frame.shape[1] - 300, 40), (frame.shape[1], 400), (255, 255, 255), 2)
            cv.imshow("data", frame)
            cv.imshow("ROI", frame[40:400, frame.shape[1] - 300:frame.shape[1]])
            frame = frame[40:400, frame.shape[1] - 300:frame.shape[1]]

        number = select_num(interrupt)
        if number == -2:
            cv.imwrite(directory + 'others/' + ('L_' if left else 'R_') + str(count['others'][('l' if left else 'r')]) + '.png',
                       frame)
        elif number >= 0:
            number = str(number)
            cv.imwrite(directory + number + '/' + ('L_' if left else 'R_') + str(count[number][('l' if left else 'r')]) + '.png',
                       frame)

    cap.release()
    cv.destroyAllWindows()


def select_num(key):
    number = -1
    if key & 0xFF == ord('o'):
        return -2
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    return number


def rename_files():
    direc = 'Image/'
    for path in ['0', '1', '2', '3', '4', '5', '6', 'others']:
        # Change this to the path of your files
        path = direc + path

        # Get a list of files that end with .png
        files_R = [f for f in os.listdir(path) if f.startswith("R")]
        files_L = [f for f in os.listdir(path) if f.startswith("L")]

        # Sort the files by their initial numbers
        files_R = sorted(files_R, key=lambda f: (int(f.split("_")[1].replace('.png', ''))))
        files_L = sorted(files_L, key=lambda f: (int(f.split("_")[1].replace('.png', ''))))
        files = [files_R, files_L]

        # Loop through the files and rename them
        for type_h in files:
            for i, file in enumerate(type_h):
                # Extract the postfix from the file name
                prefix = re.search("L_|R_", file).group()
                # Create the new file name with the index and the postfix
                new_name = prefix + str(i) + '.png'
                # Rename the file using the os.rename function
                os.rename(os.path.join(path, file), os.path.join(path, new_name))


if __name__ == '__main__':
    main()
    # rename_files()
