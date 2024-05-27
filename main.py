import os
import cv2

DATA_DIRECTORY = "dataset/Training"
GENDERS = ["male", "female"]

for gender in GENDERS:
    path = os.path.join(DATA_DIRECTORY, gender)

    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
