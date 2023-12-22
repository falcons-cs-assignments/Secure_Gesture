# Secure Gesture Project

Secure Gesture is an AI-powered system designed for face verification and hand gesture recognition. The project integrates webcam functionalities to authenticate users via facial recognition and enable control over laptop operations using hand gestures.

## Project Structure

The project directory is organized as follows:

- **face_verification:** Includes modules and scripts responsible for face verification.
- **hand_gesture:** Contains modules and scripts related to hand gesture recognition.
- **app.py:** Main application file integrating face verification and hand gesture recognition.

## Dependencies

- opencv
- numpy
- matplotlib
- tensorflow
- mtcnn
- keras-facenet
- sklearn
- pydot
- pyautogui
- mediapipe
- pandas
- seaborn

## Usage

1. **Face Verification:**
   - Run ```collect_images.py``` to collect images using webcam used for training you model on you.
   - You should get a new folder called ```photos``` with nested folder named as your user_name submited when running the previous step.
   - Create a folder called ```Unkown``` inside ```photos``` folder and copy some photos from [Labled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/) dataset here, this will be used for training model on unkown persons. Note: don't copy all ```lfw``` dataset, just some random samples may be 300 is enough.
   - If you want you model to verify other persons beside your user_name you could run ```collect_images.py``` again with different user_names.
   - Follow instructions on ```main.ipynb``` notebook in order to train your model to classify your users by thier folder names inside photos dir.
   - After training your model you can run ```verification.py``` to see if the model can recognize and verify you.

2. **Hand Gesture Control:**
   - Once verified, perform hand gestures in front of the webcam to control laptop operations.
   - Supported gestures:
     - Adjust volume
     - Take screenshots
     - Open Text Editor
     - Painter
