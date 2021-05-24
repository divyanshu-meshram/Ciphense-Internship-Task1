# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:37:57 2021

@author: hpw
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
#import cvlib as cv
from model import FacialExpressionModel
from datetime import date,datetime
import tensorflow as tf


from tensorflow.keras.models import model_from_json
#Face recognisation Model
json_file = open('C:/Users/hpw/keras-facenet/Facenet_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_FR = model_from_json(loaded_model_json)
model_FR.load_weights('C:/Users/hpw/keras-facenet/Facenet_model.h5')
FRmodel = model_FR
facec = cv2.CascadeClassifier('D:\\download\expression_recognition\haarcascade_frontalface_default.xml')

#Age model
age_model=cv2.dnn.readNetFromCaffe("C:/Users/hpw/age.prototxt.prototxt","C:/Users/hpw/dex_chalearn_iccv2015.caffemodel")
indexes=np.array([i for i in range(0,101)])

#gender Model
model_gender = load_model('C:/Users/hpw/gender_weights2.h5')

#expression Model


model_expression= FacialExpressionModel("D:\\download\model.json", "D:\\download\model_weight.h5")
classes_gender = ['man','women']
FRmodel = model_FR

def img_to_encoding(face, model):
    img = tf.keras.preprocessing.image.load_img(face, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

database = {}
database["divyanshu"] = img_to_encoding("C:/Users/hpw/training_images/divyanshu.jpg", FRmodel)
database["manoj"] = img_to_encoding("C:/Users/hpw/training_images/manoj.jpg", FRmodel)
database["priyamani"] = img_to_encoding("C:/Users/hpw/training_images/priyamani.jpg", FRmodel)

def verify(encoding,model,database):
    l=[]
    for i in database:
        l.append(i)
        
    #encoding = img_to_encoding(face, model)
    dist=[]
    for i in database:
        dist.append(np.linalg.norm(encoding - database[i]))
    idx=np.argmin(dist)
    m=min(dist)
    a=l[idx]
    return a
    """
    if m<1:
        a=l[idx]
    else:
        a="not found"
    return a,m
"""

webcam = cv2.VideoCapture(0)
#webcam = cv2.VideoCapture('D:/download/expression_recognition/videoplayback.mp4')
while webcam.isOpened():
    status, frame = webcam.read()
    #face, confidence = cv.detectface(frame)
    #faces1=detector.detectMultiScale(frame,1.3,5)
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    now = datetime.now()

    
        
    today = date.today()

    # dd/mm/YY
    
    for (x, y, w, h) in faces:
        
        detected_face=frame[int(y):int(y+h),int(x):int(x+w)]
        #for age detection
        detected_face=cv2.resize(detected_face,(224,224))
        detected_face_blob=cv2.dnn.blobFromImage(detected_face)
        age_model.setInput(detected_face_blob)
        age_result=age_model.forward()
        a=int(round(np.sum(age_result[0]*indexes)))
        a=str(a)
        
        #for face recognisation
        img=cv2.resize(detected_face,(160,160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = FRmodel.predict_on_batch(x_train)
        embedding=embedding / np.linalg.norm(embedding, ord=2)
        b=verify(embedding,FRmodel,database)
        
        #for gender recognisation
        detected_face_2=cv2.resize(detected_face,(96,96))
        face_crop = img_to_array(detected_face_2)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = model_gender.predict(face_crop)[0]
        idx = np.argmax(conf)
        label_gender = classes_gender[idx]
        
        #for expression recognisation
        
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        predicted_expression = model_expression.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        
        
        
        #print(faces.shape)
        
        # write label and confidence above face rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, b, (x,y-20),font,0.7, (0, 0, 255), 2)
        cv2.putText(frame, a, (x,y+h+40),font,0.7, (0, 0, 255), 2)
        cv2.putText(frame, label_gender, (x,y+h+20),font,
                    0.7, (0, 0, 255), 2)
        cv2.putText(frame, predicted_expression, (x,y+h+60), font, 0.7, (0, 0, 255), 2)
        current_time = now.strftime("%H:%M:%S")
        d1 = today.strftime("%d/%m/%Y")
        cv2.putText(frame,current_time, (x,y+h+80), font,0.5 ,(0, 0, 255), 1)
        cv2.putText(frame,d1, (x,y+h+100), font, 0.5, (0, 0, 255),1)
        # display output
        cv2.putText(frame,"Number of face detected "+str(faces.shape[0]),(0,25),font,0.7,(0,0,255),2)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release resources
webcam.release()
cv2.destroyAllWindows()