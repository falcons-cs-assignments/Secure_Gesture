import cv2 as cv
import math
import platform
import numpy as np
import pyautogui
import subprocess
from datetime import datetime
import time
import psutil
import os

if platform.system() == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
else:
    from subprocess import call



folder_path = 'screenshots'
def take_screenshot():
    dt_string = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pyautogui.screenshot(f'screenshots/Screenshot{dt_string}.png')
    time.sleep(2)


def open_application(application):
    """
    application: settings or text editor
    It used to open only settings or text editor on any Operating System
       windows:
               Text editor: r'C:\Windows\System32\notepad.exe'
               Settings   : control.exe

        Linux:
               Text editor: gedit
               Settings   : gnome-control-center

    """

    if platform.system() == "Windows":
        if application.lower() == 'text editor':
            application = r'C:\Windows\System32\notepad.exe'
        if application.lower() == 'settings':
            application = 'control.exe'
        subprocess.Popen(application)

    elif platform.system() == "Linux":
        if application.lower() == 'text editor':
            application = 'gedit'
        if application.lower() == 'settings':
            application = 'gnome-control-center'
        subprocess.Popen(application, shell=True)

    else:
        print("Unsupported operating system.")

    time.sleep(2)
        

# def terminate_process(process_name):
#     for process in psutil.process_iter(['pid', 'name']):
#         if process.info['name'] == process_name:
#             try:
#                 pid = process.info['pid']
#                 os.system(f"kill {pid}")  # Sending a signal to terminate the process
#                 print(f"Process {process_name} (PID: {pid}) terminated successfully.")
#             except Exception as e:
#                 print(f"Error terminating process {process_name}: {e}")
#             return
#     print(f"Process {process_name} not found.")


class Speaker:
    def __init__(self, max_distance=0.55):
        """
            distance: value in range [0.0 : 0.55]
            volume level: value in (dB) {0% --> -65.25dB,
                                        100% --> 0dB}
                                        # NOTE: -6.0 dB = half volume !
        """
        if platform.system() == "Windows":
            # Get default audio device using PyCAW
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))

            self.distance_range = [0, max_distance]
            self.vol_range_dB = self.volume.GetVolumeRange()[:2]     # [-65.25, 0.0]
        # print(self.vol_range_dB)

    def __call__(self, distance):
        # map distance into volume level
        # TODO: make mapping is logarithmic as it for dB :)
        distance = min(distance, 0.55)

        if platform.system() == "Linux":
            cur_vol = np.interp(distance, [0.1, 0.55], [0, 100])
            call(["amixer", "-D", "pulse", "sset", "Master", f"{cur_vol}%"])

        elif platform.system() == "Windows":
            ratio = distance / self.distance_range[1]
            level_dB = 28 * math.log(max(0.0001, ratio))   # level = ln(ratio)

            # Get current volume
            self.volume.SetMasterVolumeLevel(max(level_dB, self.vol_range_dB[0]), None)
            current_volume_dB = self.volume.GetMasterVolumeLevel()

            cur_vol = math.exp(current_volume_dB / 15) * 100

        return cur_vol


class Painter:
    def __init__(self, height, width, folder_path="Header", brush_thickness=5,
                 eraser_thickness=100, draw_color=(0, 0, 255)):

        self.height = height
        self.width = width
        self.folder_path = folder_path
        self.brushThickness = brush_thickness
        self.eraserThickness = eraser_thickness
        self.drawColor = draw_color
        self.xp, self.yp = 0, 0

        self.imgCanvas = np.zeros((height, width, 3), np.uint8)

        myList = os.listdir(folder_path)
        print(myList)
        self.overlayList = []
        for imPath in myList:
            image = cv.imread(f'{folder_path}/{imPath}')
            self.overlayList.append(image)
        print(len(self.overlayList))
        self.header = self.overlayList[0]

    def paint(self, img):
        self.imgGray = cv.cvtColor(self.imgCanvas, cv.COLOR_BGR2GRAY)
        _, self.imgInv = cv.threshold(self.imgGray, 0, 255, cv.THRESH_BINARY_INV)
        self.imgInv = cv.cvtColor(self.imgInv, cv.COLOR_GRAY2BGR)
        img = cv.bitwise_and(img, self.imgInv)
        img = cv.bitwise_or(img, self.imgCanvas)

        return img

    def selection(self, img, x1, y1):
        w_ratio = self.width/5
        r = 0.8552
        color_w = r * int(self.height/6) 

        self.xp, self.yp = 0, 0
        # print("Selection Mode")
        # # Checking for the click
        if y1 < self.height/6:
            if w_ratio < x1 < w_ratio+color_w:
                self.header = self.overlayList[0]
                self.drawColor = (0, 0, 255)

            elif w_ratio*2 < x1 < (w_ratio*2)+color_w:
                self.header = self.overlayList[1]
                self.drawColor = (255, 0, 0)

            elif w_ratio*3 < x1 < (w_ratio*3)+color_w:
                self.header = self.overlayList[2]
                self.drawColor = (0, 255, 0)

            elif w_ratio*4 < x1 < self.width:
                self.header = self.overlayList[3]
                self.drawColor = (0, 0, 0)

        cv.rectangle(img, (x1-10, y1 - 10), (x1+10, y1 + 10), self.drawColor, cv.FILLED)

        return img

    def drawing(self, img, x1, y1):
        cv.circle(img, (x1, y1), 5, self.drawColor, cv.FILLED)
        # print("Drawing Mode")
        # if hand_flag:
        #     self.xp, self.yp = 0, 0
        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = x1, y1

        if self.drawColor == (0, 0, 0):
            cv.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
            cv.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)

        else:
            cv.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)
            cv.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)

        self.xp, self.yp = x1, y1

        return img

    def show_header(self, img):
        # Setting the header image
        self.header = cv.resize(self.header, (self.width, int(self.height/6)))
        img[0:int(self.height/6), 0:self.width] = self.header
        
        return img

    def show_windows(self):
        cv.imshow("Canvas", self.imgCanvas)
        cv.imshow("gray", self.imgGray)
        cv.imshow("Inv", self.imgInv)


def main():
    volume = Speaker()
    print(volume(0.1))


if __name__ == "__main__":
    main()
