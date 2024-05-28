import os
import cv2
from random import shuffle

DATA_DIRECTORY = "dataset/Training"
GENDERS = ["male", "female"]

training_data = []

for gender_index, gender in enumerate(GENDERS):
    path = os.path.join(DATA_DIRECTORY, gender)

    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        training_data.append([image_array, gender_index])

shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
