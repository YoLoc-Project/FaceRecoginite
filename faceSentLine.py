import requests
import cv2
from face_recognition import FaceRecognition
import base64
from face_recognition import FaceRecognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
import sys
import cv2
import base64
from tqdm import tqdm
import requests
from pprint import pprint
import shutil
import time
from parinya import LINE
import RPi.GPIO as GPIO
from pathlib import Path

#setup default relay and solenoid
RELAY = 23
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
#GPIO.output(RELAY,GPIO.LOW)
GPIO.output(RELAY,0)
######

MODEL_PATH = "model_v1.pkl"

ROOT_FOLDER ="./FirebaseDatasets/"

# Initialize the FaceRecognition class
fr = FaceRecognition()  

# Load the saved model
fr.load(MODEL_PATH)

# TEMP NAME LIST
# TODO: Grab the name (or email) list

# Line Notify API URL
LINE_NOTIFY_API_URL = "https://notify-api.line.me/api/notify"

# Your Line token
LINE_TOKEN = "l4J4P1h95XZRQ9WQIOfgYY8OIAAnRKcOMrwUIJWFbMF"
line = LINE('l4J4P1h95XZRQ9WQIOfgYY8OIAAnRKcOMrwUIJWFbMF')

# Start video capture
cap = cv2.VideoCapture(-1)

while True:
    start_time = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # line.sendimage(frame[:, :, ::-1])
   
    time.sleep(30)
    

    # if ret:
    current_time = time.time()
    if current_time - start_time >= 30: 
        # Use the FaceRecognition class to make predictions on the frame

        result = fr.predict(frame, threshold=0.6)

        if len(result['predictions']) > 0:
            print("Image predicted")

            # Get the name of the person recognized in the frame
            name = result['predictions'][0]['person']
            confidence = result['predictions'][0]['confidence']

            # Convert the frame to a base64-encoded string
            frame_b64 = base64.b64encode(frame).decode()

            # Send the results to Line
            headers = {
                "Authorization": "Bearer " + LINE_TOKEN,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            payload = {
                "message": f"Person recognized: {name}",
                "imageFile": frame_b64
            }
            line.sendimage(frame[:, :, ::-1])
            response = requests.post(LINE_NOTIFY_API_URL, headers=headers, data=payload)
            pprint(response.json())


            newlist = []
            list = os.listdir(ROOT_FOLDER)

            for folder in list:
                newlist.append(folder)

            #if match data person unlock door
            if name in newlist:
                
                GPIO.output(RELAY,1) # trigger relay 
                print("door unlock")
                time.sleep(5)
               
                GPIO.output(RELAY,0)
                print("door lock")

            # current_time = time.time()
            # if current_time - start_time >= 30:
            #     print("Time's up")
            #     break

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Turning off")
        break
    

# Release the video capture
cap.release()
cv2.destroyAllWindows()
   