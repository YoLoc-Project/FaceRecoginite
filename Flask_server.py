from flask import Flask, redirect, url_for, request, jsonify, Response
from face_recognition import FaceRecognition
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

# FLASK SERVER
# --------------------------------------------
port = 5000
app = Flask(__name__)

@app.route('/flask/test', methods=['GET','POST'])
def test():
   if request.method == 'POST':
      print('POST')
      mydata = request.__dict__
      print(mydata)
      return "Flask server post"
   else:
      print('GET')
      return "Flask server get"
   
@app.route('/flask/trainmodel', methods=['POST'])
def RetrainModel():

   mydata = request.__dict__
   newdata_string = request.get_json()
   # print(newdata_string)
   newdata = json.loads(newdata_string)
   newFaces = newdata['faceImgs']
   email = newdata['email']
   nickname = newdata['nickname']
   # print(newFaces)

   train_dataset = []

   # Create a folder for the user
   userPath = ROOT_FOLDER + email + '/'
   if not os.path.exists(userPath):
      os.makedirs(userPath)

   # Get image from given firebase urls
   for i, url in enumerate(newFaces):
      img_data = requests.get(url).content
      # img_data = requests.get("https://firebasestorage.googleapis.com/v0/b/yoloc-face-recognition.appspot.com/o/faces%2FIMG_5635.jpg?alt=media&token=d34bbfdd-a70f-470b-809b-cf44e568a1f7").content
      destination = os.path.join(userPath, nickname + '_' + str(i) + '.jpg')
      print(destination)
      with open(destination, 'wb') as handler:
         handler.write(img_data)

   # insert new dataset

   # append dataset
   for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
      path = path.replace("\\","/")
      email = path.split("/")[-2]
      train_dataset.append({"person" :email, "path": path})

   
   if len(train_dataset) <= 0:
      return Response("Flask server error", status=500)
   
   train_dataset = pd.DataFrame(train_dataset)
   train_dataset = train_dataset.groupby("person").filter(lambda x: len(x) >= 1)

   # train_dataset.head(20)
   # Training
   fr = FaceRecognition()
   fr.fit_from_dataframe(train_dataset)
   fr.save(MODEL_PATH)
   return "Dataset has been retrained"
# --------------------------------------------

if __name__ == "__main__":

    print("running flask server")
    app.run(port=port, debug=True, threaded=True) # threaded=True is for serving multiple clients

    

