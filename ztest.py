import os
import glob

ROOT_FOLDER ="./FirebaseDatasets/"
list = os.listdir(ROOT_FOLDER)

newlist = []

for folder in list:
    newlist.append(folder)

print(type(folder))
print(newlist)