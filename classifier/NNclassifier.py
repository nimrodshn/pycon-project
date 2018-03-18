#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import argparse
import cv2
import numpy as np
import random

random.seed()

parser = argparse.ArgumentParser(description='A learning line Classifier.')
parser.add_argument('image', metavar='FILE', type=str,
                    help='image file name to proccess')
parser.add_argument('sampels', metavar='N', type=int, nargs='?', default=360,
                    help='number of positive training points')
args = parser.parse_args()

# get number of sumples from user
num_of_samples = args.sampels

# get imagename from user
img_name = args.image
input_image = cv2.imread('pictures/' + img_name)
output_image = cv2.imread('pictures/out/' + img_name)

width, length, _ = input_image.shape
training_data = []
training_label = []

print("Create some training data.")
training_length = 0

while training_length < num_of_samples:
    # randomly chose point
    x, y = random.randint(0, width - 33), random.randint(0, length - 33)

    # append label
    output_scalar = 0.99 if output_image[x+16, y+16][0] > 0 else 0.01
    training_label.append(output_scalar)

    # append training data
    input_matrix = input_image[x:(x+32), y:(y+32)]
    training_data.append(input_matrix)

    training_length = training_length + output_scalar

training_data = np.array(training_data, dtype=np.uint8)
#training_label = np.array(training_label, dtype=np.uint8)
print(training_data.shape)
#print(training_label.shape)

def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=training_data[0].shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    return model


print("Start training on data.")

model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(training_data, training_label,
                    batch_size=batch_size, epochs=epochs, verbose=2)
