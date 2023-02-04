import os
import cv2
import numpy as np
from music import *
from keras.models import model_from_json
# from keras.preprocessing import image
import keras.utils as image
from keras.utils import load_img, img_to_array

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
    # captures frame and returns boolean and captured image
    ret, img_test = cam.read()
    if not ret:
        continue
    #convert to gray image and resize
    gray_img = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img_test, (x,y), (x+w, y+h),(255,200,255),thickness=7)
        roi_gray = gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pix = image.img_to_array(roi_gray)
        img_pix = np.expand_dims(img_pix, axis = 0)
        img_pix /= 255
        pred = model.predict(img_pix)
        max_ind = np.argmax(pred[0]) #max prediction
        emo = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        pred_emo = emo[max_ind]
        song = recommend(pred_emo)
        cv2.putText(img_test, pred_emo,  (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (160,100,255), 2)
        cv2.putText(img_test, song,  (int(x + 30), int(y + 30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (153,0,76), 2)
    
    crop_img = cv2.resize(img_test, (1000,700))
    cv2.imshow('Facial emotion analysis ', crop_img)
    
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cam.release()
cv2.destroyAllWindows