import numpy as np
import cv2
import os
import random
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

data_dir=r'C:\Users\Pratik\Desktop\face_detection\face_dataset\train'
categories=["pratik","not.pratik"]
x=[]
y=[]

for category in categories:
  path=os.path.join(data_dir,category)
  class_num=categories.index(category)
  for img in os.listdir(path):
    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(150,150))
    x.append(new_array)
    y.append(class_num)
    
c = list(zip(x, y))
random.shuffle(c)
x, y = zip(*c)

x=np.array(x).reshape(-1,150,150,1)
x=x/255
x=np.array(x)
y=np.array(y)

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

history=model.fit(x,y,batch_size=32,epochs=2,validation_split=0.1)

model.save('face_model.h5')
