from face_recognition import FaceRecognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import csv
import time
# import random
# import numpy as np
import cv2
# import base64
from tqdm import tqdm
# import requests
from pprint import pprint

#Settings
IMAGE_PATH ="./Datasets/Test1/"
MODEL_PATH = "model_v1.pkl"
RESULT_PATH = "Test1_Result.csv"
THRESHOLD = 0.6

#Read Test dataset
test_dataset = []
for path in glob.iglob(os.path.join(IMAGE_PATH, "**", "*.jpg")):
    path = path.replace("\\","/")
    person = path.split("/")[-2]
    test_dataset.append({"person":person, "path": path})
    
test_dataset = pd.DataFrame(test_dataset)
if test_dataset.empty:
    print("Test dataset is empty!")
    exit()
    
# Must include at least x faces per person
test_dataset = test_dataset.groupby("person").filter(lambda x: len(x) >= 1)

# Testing
fr = FaceRecognition()
fr.load(MODEL_PATH)

# Starts total timer
start_timer = time.time()

#Create a new csv result file
with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as file:
    # create the csv writer
    writer = csv.writer(file)
    writer.writerow(["Input", "Result", "Time", "Confidence"])

    y_test, y_pred, y_scores = [],[],[]
    for idx in tqdm(range(len(test_dataset))):
        path = test_dataset.path.iloc[idx]
        img =  cv2.imread(path)
        person = path.split("/")[-2]

        result = fr.predict(img, threshold=THRESHOLD)
        for prediction in result["predictions"]:
            print(result["predictions"])
            y_pred.append(prediction["person"])
            y_scores.append(prediction["confidence"])
            print("INPUT : ", person)
            print("RESULT: ", prediction["person"])
            print("TIME  : %f seconds" % result["elapsed_time"])
            print("CONFIDENCE : %f" %  prediction["confidence"])
            writer.writerow([person, prediction["person"], result["elapsed_time"], prediction["confidence"]])
            y_test.append(test_dataset.person.iloc[idx])

# Show Summary
print(classification_report(y_test, y_pred))

print("Unique faces: ", test_dataset["person"].nunique())
print("Tested image count: ", test_dataset.shape[0])
print("Time Taken: %f seconds" % (time.time() - start_timer))

# Accuracy  
print("Accuracy: %f" % accuracy_score(y_test, y_pred))