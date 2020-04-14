import numpy as np
import cv2
import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model


img=cv2.imread(r'F:\python\Deep learning\face_detection\pratik_7.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('img',img)
model = tf.keras.models.load_model('face_model.h5')
img=cv2.resize(img,(150,150))
img=np.array(img).reshape(-1,150,150,1)
result=model.predict_classes(img)
print(result)
