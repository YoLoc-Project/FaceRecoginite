# from flask import Flask, redirect, url_for, request, jsonify, Response
from face_recognition import FaceRecognition
import socketio
import os
import glob
import pandas as pd
import requests
import json

#Settings
ROOT_FOLDER ="./FirebaseDatasets/"
MODEL_PATH = "model_v1.pkl"

# Folder
if not os.path.exists(ROOT_FOLDER):
    os.makedirs(ROOT_FOLDER)

# CONNECT TO THE BACKEND SERVER HERE
# --------------------------------------------
# port = "http://localhost:3000"
port = 'https://yoloc-backend.herokuapp.com/'
sio = socketio.Client()
sio.connect(port)

print("Starting client and connecting to backend port ", port)

@sio.event
def connect():
    print("Connected to the backend server at", port)

@sio.event
def disconnect():
    print("Disconnected from the backend server")

# sio.emit('message', {'from': 'client'}

@sio.on('test')
def test(data):
   print(data)
   sio.emit("client_success", {'statusCode': 200, 'message': 'Success'})
   
@sio.on('trainmodel')
def RetrainModel(request):

   mydata = request['json']
   # newdata_string = request.get_json()
   # print(newdata_string)
   newdata = json.loads(mydata)
   newFaces = newdata['faceImgs']
   email = newdata['email']
   # nickname = newdata['nickname']

   train_dataset = []

   # Create a folder for the new user
   userPath = ROOT_FOLDER + email + '/'
   if not os.path.exists(userPath):
      os.makedirs(userPath)

   # Get image from given firebase urls
   for i, url in enumerate(newFaces):
      img_data = requests.get(url).content
      # img_data = requests.get("https://firebasestorage.googleapis.com/v0/b/yoloc-face-recognition.appspot.com/o/faces%2FIMG_5635.jpg?alt=media&token=d34bbfdd-a70f-470b-809b-cf44e568a1f7").content
      destination = os.path.join(userPath, 'IMG' + '_' + str(i) + '.jpg')
      print(destination)
      with open(destination, 'wb') as handler:
         handler.write(img_data)

   # append dataset
   for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
      path = path.replace("\\","/")
      email = path.split("/")[-2]
      train_dataset.append({"person" :email, "path": path})

   
   if len(train_dataset) <= 0:
      # Can't seem to do response-style output yet
      sio.emit("client_error", {'statusCode': 400, 'message': 'Client error due to insufficient dataset'})
      return
   
   train_dataset = pd.DataFrame(train_dataset)
   train_dataset = train_dataset.groupby("person").filter(lambda x: len(x) >= 1)

   # train_dataset.head(20)
   # Training
   
   fr = FaceRecognition()
   fr.fit_from_dataframe(train_dataset)
   fr.save(MODEL_PATH)
   print("Image training completed")
   sio.emit("client_success", {'statusCode': 200, 'message': 'Successfully trained dataset'})

   return
# --------------------------------------------
