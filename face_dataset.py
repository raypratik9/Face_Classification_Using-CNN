import numpy as np
import cv2
import os
path=r'F:\python\Deep learning\face_detection\face_dataset\train\not.pratik'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
c=0
while 1:
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        c=c+1
        cv2.imshow('img2',img[y:y+h,x:x+w])
        cv2.imwrite(os.path.join(path , 'dd_'+str(c)+'.jpg'),img[y:y+h,x:x+w].copy())
    cv2.imshow('img',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    if (c>=500):
        break
cap.release()
cv2.destroyAllWindows()
    
