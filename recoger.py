import cv2
import numpy as np
import os
from tensorflow import keras
from keras.models import model_from_json

cap = cv2.VideoCapture(0)

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file) 

classes = {0:'Angry', 
           1:'Disgust',
           2:'Fear',
           3:'Happy',
           4:'Sad',
           5:'Surprise',
           6:'Neutral'}

model = keras.models.load_model('model_emotion_recog.h5')

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    faces = face_cascade.detectMultiScale(frame, 1.3, 4) 

    for (x, y, w, h) in faces: 
        img = cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (255, 0, 0), 2)
        req = img[y:y+h,x:x+w]
        gray_req = cv2.cvtColor(req, cv2.COLOR_BGR2GRAY)
        resized_req = cv2.resize(gray_req, (48,48),interpolation = cv2.INTER_NEAREST)
        reshaped = np.reshape(resized_req,[1,48,48,1])
        prediction = model.predict(reshaped)
        ans = np.argmax(prediction[0])
        text = cv2.putText(frame, classes[ans], (x-20, y-20), font,  #classes[ans]
                   2, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()