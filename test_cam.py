import numpy as np
import cv2
import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model =tf.keras.models.load_model('face_model.h5')
cap = cv2.VideoCapture(1)
while 1:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img1=img[y:y+h,x:x+w].copy()
        cv2.imshow('img1',img1)
        img1=cv2.resize(img1,(150,150))
        cv2.imshow('img2',img1)
        test_img=image.img_to_array(img1)
        test_img=np.expand_dims(test_img,axis=0)
        result=model.predict_classes(test_img)
        print(result)
        if(result==0):
            cv2.putText(img,'pratik',(150, 150) ,cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0) , 2, cv2.LINE_AA)
        else:
            cv2.putText(img,'not_pratik',(150, 150) ,cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0) , 2, cv2.LINE_AA)
        
    cv2.imshow('img',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
