import os
import cv2
from random import shuffle
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

DATA_DIRECTORY = "dataset/Training"
GENDERS = ["male", "female"]

IMAGE_SIZE = 100
training_data = []

for gender_index, gender in enumerate(GENDERS):
    path = os.path.join(DATA_DIRECTORY, gender)

    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([image_array, gender_index])

shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = numpy.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
y = numpy.array(y)

X = X / 255

model = Sequential([
    Conv2D(64, (3,3), input_shape=X.shape[1:], activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(64),
    
    Dense(1, activation="sigmoid"),
    ])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=5, validation_split=0.1)

while True:
    image_path = input("Enter the path to an image of a man or woman.\n")
    
    if os.path.isfile(image_path):
        break
    
image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))

prediction = model.predict([image_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)])
print(f"The person in the image you provided is a {GENDERS[int(prediction[0][0])]}.")
