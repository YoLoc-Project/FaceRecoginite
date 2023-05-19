from face_recognition import FaceRecognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import os
# import glob
# import pandas as pd
# import random
# from tqdm import tqdm
# import requests

import numpy as np
import sys
import cv2
import base64
from pprint import pprint
import shutil
from multiprocessing import Process

MODEL_PATH = "model_v1.pkl"

def fr():
    print("flask")

SAVE_PATH = "Result"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
# else:    
#     shutil.rmtree(SAVE_PATH)
#     os.mkdir(SAVE_PATH)
SAVE_PATH = os.path.join(os.getcwd(), SAVE_PATH)

# Testing
fr = FaceRecognition()
fr.load(MODEL_PATH)

def webcam_face_detect(video_mode, nogui = False, cascasdepath = "haarcascade_frontalface_alt.xml"):

    face_cascade = cv2.CascadeClassifier(cascasdepath)

    video_capture = cv2.VideoCapture(video_mode)
    num_faces = 0


    while True:
        ret, image = video_capture.read()

        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 3,
            minSize = (30,30)

            )

        # print("The number of faces found = ", len(faces))
        num_faces = len(faces)

        if num_faces > 0:
            result = fr.predict(image, threshold=0.5)
            file_bytes = np.fromstring(base64.b64decode(result["frame"]), np.uint8)
            output = cv2.imdecode(file_bytes,1)

            plt.imshow(output)
            if len(result['predictions'])>0:
                # Show the first face recognition results
                plt.title("%s (%f)" % (result["predictions"][0]["person"], result["predictions"][0]["confidence"]))
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            pprint(result["predictions"])


        if not nogui:
            for (x,y,w,h) in faces:
                cv2.rectangle(image, (x,y), (x+h, y+h), (0, 255, 0), 2)

            cv2.imshow("Face recog", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
    return num_faces

if __name__ == "__main__":

    if len(sys.argv) < 2:
        video_mode = 0
    else:
        video_mode = sys.argv[1]

    webcam_face_detect(video_mode)

    # webcamThread = Process(target=webcam_face_detect(video_mode))
    # webcamThread.start()

    # serverThread = Process(target=fr)
    # serverThread.start()

    # webcamThread.join()
    # serverThread.join()

    # print("running flask server")
    # app.run(port=port, debug=True, threaded=True) # threaded=True is for serving multiple clients

    

