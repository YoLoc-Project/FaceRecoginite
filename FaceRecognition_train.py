from face_recognition import FaceRecognition
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
# import matplotlib.pyplot as plt
import os
import glob
import time
import pandas as pd
# import random
# import numpy as np
# import cv2
# import base64
# from tqdm import tqdm
# import requests
# from pprint import pprint

#Settings
ROOT_FOLDER ="./Datasets/Train/"
MODEL_PATH = "model_v1.pkl"

#Read Train dataset
train_dataset = []
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
    path = path.replace("\\","/")
    person = path.split("/")[-2]
    train_dataset.append({"person" :person, "path": path})
    
train_dataset = pd.DataFrame(train_dataset)

if train_dataset.empty:
    print("Train dataset is empty!")
    exit()

# Must include at least x faces per person
train_dataset = train_dataset.groupby("person").filter(lambda x: len(x) >= 1)

start_timer = time.time()

# Training
print("Begin face training.")
fr = FaceRecognition()
fr.fit_from_dataframe(train_dataset)
fr.save(MODEL_PATH)

print("Unique faces: ", train_dataset["person"].nunique())
print("Trained image count: ", train_dataset.shape[0])
print("Time Taken: %f seconds" % (time.time() - start_timer))

