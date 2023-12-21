import cv2
import os
from datetime import datetime

# Ask for the user's name
user_name = input("Please enter your name: ")

# Create a directory for the user inside the photos directory
os.makedirs(f'photos/{user_name}', exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Check for keypress
    k = cv2.waitKey(1)

    # If space bar is pressed, save the image
    if k%256 == 32:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"photos/{user_name}/{timestamp}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")

    # If 'ESC' is pressed, quit
    elif k%256 == 27:
        print("Escape hit, closing...")
        break

# When everything done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()

