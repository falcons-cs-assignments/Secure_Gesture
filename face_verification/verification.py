import cv2 as cv
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress Tensorflow logging messages
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet


class verification:
    def __init__(self, path='./'):
        self.facenet = FaceNet()
        self.faces_embeddings = np.load(path + "faces_embeddings_done_4classes.npz")
        Y = self.faces_embeddings['arr_1']
        self.encoder = LabelEncoder()
        self.encoder.fit(Y)
        self.haarcascade = cv.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
        self.model = pickle.load(open(path + "svm_model_160x160.pkl", 'rb'))

    def verify(self, frame, draw=True):
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)

        for x, y, w, h in faces:
            img = rgb_img[y:y + h, x:x + w]
            img = cv.resize(img, (160, 160))  # 1x160x160x3
            img = np.expand_dims(img, axis=0)
            ypred = self.facenet.embeddings(img)
            face_name = self.model.predict(ypred)
            final_name = self.encoder.inverse_transform(face_name)[0]
            if draw:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
                cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 3, cv.LINE_AA)

            if final_name == "others":
                return False
            else:
                return True

    def rea_time_verify(self):
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            _, frame = cap.read()
            frame = cv.flip(frame, 1)
            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)
            for x, y, w, h in faces:
                img = rgb_img[y:y + h, x:x + w]
                img = cv.resize(img, (160, 160))  # 1x160x160x3
                img = np.expand_dims(img, axis=0)
                ypred = self.facenet.embeddings(img)
                face_name = self.model.predict(ypred)
                final_name = self.encoder.inverse_transform(face_name)[0]
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
                cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 3, cv.LINE_AA)

            cv.imshow("Face Recognition:", frame)

            key = cv.waitKey(1)
            if key == 27:  # Esc
                break

        cap.release()
        cv.destroyAllWindows


if __name__ == "__main__":
    verifier = verification()
    verifier.rea_time_verify()
    