from face_recognition import FaceRecognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
import cv2
import base64
from tqdm import tqdm
import requests
from pprint import pprint
import shutil

#Settings
IMAGE_PATH ="./Datasets/Test2/"
MODEL_PATH = "model_v1.pkl"

SAVE_PATH = "Result"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
else:    
    shutil.rmtree(SAVE_PATH)
    os.mkdir(SAVE_PATH)
SAVE_PATH = os.path.join(os.getcwd(), SAVE_PATH)

# Testing
fr = FaceRecognition()
fr.load(MODEL_PATH)

file_names = [fn for fn in os.listdir(IMAGE_PATH) if any(fn.endswith(ext) for ext in ['jpg','jpeg', 'bmp', 'png', 'gif'])]

for file in file_names:
    img =  cv2.imread(os.path.join(IMAGE_PATH,file))

    result = fr.predict(img, threshold=0.6)
    file_bytes = np.fromstring(base64.b64decode(result["frame"]), np.uint8)
    output = cv2.imdecode(file_bytes,1)

    plt.imshow(output)
    if len(result['predictions'])>0:
        # Show the first face recognition results
        plt.title("%s (%f)" % (result["predictions"][0]["person"], result["predictions"][0]["confidence"]))
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    plt.savefig(os.path.join(SAVE_PATH,'Output_'+file))
    plt.close()
    pprint(result["predictions"])
