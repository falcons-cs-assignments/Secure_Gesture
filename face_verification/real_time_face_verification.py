import os
import shutil
import random
import numpy as np
import cv2
import tensorflow as tf
from layers import L1Dist


def init_app_data():
    # create app_data folder which will store input_image, verification_images
    INPUT_IMG_PATH = os.path.join('app_data', 'input_image')
    VERIFICATION_IMGS_PATH = os.path.join('app_data', 'verification_images')
    os.makedirs(INPUT_IMG_PATH, exist_ok=True)
    os.makedirs(VERIFICATION_IMGS_PATH, exist_ok=True)

    # copy random subset of anchor, positive images to be used as verification_images
    # Define the paths to your anchor and positive image folders
    POSITIVE_IMGS_PATH = os.path.join('data', 'positive')

    # Get a list of all positive images
    positive_images = os.listdir(POSITIVE_IMGS_PATH)

    # Clear the verification_images folder
    for filename in os.listdir(VERIFICATION_IMGS_PATH):
        file_path = os.path.join(VERIFICATION_IMGS_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Define the number of images you want to copy
    num_images_to_copy = 50  # Adjust this value as needed

    # Randomly select a subset of positive images
    selected_positive_images = random.sample(positive_images, num_images_to_copy)

    # Copy the selected images to the verification_images folder
    for image in selected_positive_images:
        shutil.copy(os.path.join(POSITIVE_IMGS_PATH, image), VERIFICATION_IMGS_PATH)


# Function to preprocess the images
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)   # Read img
    img = tf.io.decode_jpeg(byte_img)       # Load img
    img = tf.image.resize(img, (100, 100))  # Resize
    img = img / 255.0                       # Normalization
    return img


# Function to compare input_image with verification_images and return verification
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    
    for image in os.listdir(os.path.join('app_data', 'verification_images')):
        input_img = preprocess(os.path.join('app_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('app_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('app_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


# Function to perform real-time verification
def real_time_verification(model, detection_threshold, verification_threshold):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120+250, 200:200+250, :]

        cv2.imshow('Verification', frame)

        # Store the key pressed
        key = cv2.waitKey(1) & 0xFF

        # Verification trigger
        if key == ord('v'):
            # Save input image to app_data/input_image folder
            cv2.imwrite(os.path.join('app_data', 'input_image', 'input_image.jpg'), frame)
            # Run verification
            results, verified = verify(model, detection_threshold, verification_threshold)
            print(verified)

        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def verify_frame(frame):
    # init app_data folder with verifcation_images
    init_app_data()

    # Reload model weights
    model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist':L1Dist})

    # Set the detection and verification thresholds
    detection_threshold = 0.99
    verification_threshold = 0.7

    frame = frame[120:120+250, 200:200+250, :]

    # Save frame as input_image.jpg to app_data/input_image folder
    cv2.imwrite(os.path.join('app_data', 'input_image', 'input_image.jpg'), frame)

    # Run verification
    results, verified = verify(model, detection_threshold, verification_threshold)
    print(verified)

    # Return result
    return verified


if __name__ == "__main__":
    # init app_data folder with verifcation_images
    init_app_data()

    # Reload model weights
    model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist':L1Dist})

    # Set the detection and verification thresholds
    detection_threshold = 0.99
    verification_threshold = 0.7

    # Start real-time verification
    real_time_verification(model, detection_threshold, verification_threshold)
