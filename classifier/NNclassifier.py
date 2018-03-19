#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import argparse
import cv2
import numpy as np
import random

batch_size = 256
epochs = 13
random.seed()

# parse user data
parser = argparse.ArgumentParser(description='A learning line Classifier.')
parser.add_argument('image', metavar='FILE', type=str,
                    help='image file name to proccess')
parser.add_argument('testimage', metavar='FILE', type=str,
                    help='test image file name to proccess')
parser.add_argument('sampels', metavar='N', type=int, nargs='?', default=360,
                    help='number of training points')
args = parser.parse_args()

# get number of sumples from user
num_of_samples_per_category = args.sampels / 2


def get_data(img1, img2, samples):
    width, length, _ = img1.shape
    training_data = []
    training_label = []

    training_length = 0
    while training_length < samples:
        # randomly chose point
        x, y = random.randint(0, width - 33), random.randint(0, length - 33)

        if img2[x+16, y+16][0] == 0:
            continue

        # append label
        training_label.append(0.01)

        # append training data
        input_matrix = img1[x:(x+32), y:(y+32)]
        training_data.append(input_matrix)

        training_length = training_length + 1

    training_length = 0
    while training_length < samples:
        # randomly chose point
        x, y = random.randint(0, width - 33), random.randint(0, length - 33)

        if img1[x+16, y+16][0] != 0:
            continue

        # append label
        training_label.append(0.99)

        # append training data
        input_matrix = img1[x:(x+32), y:(y+32)]
        training_data.append(input_matrix)

        training_length = training_length + 1

    training_data = np.array(training_data, dtype=np.uint8)
    training_label = np.array(training_label, dtype=np.uint8)

    return training_data, training_label


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


print("Create some training data with labels.")

input_image = cv2.imread('pictures/' + args.image)
output_image = cv2.imread('pictures/out/' + args.image)
training_data, training_label = get_data(input_image,
                                         output_image,
                                         num_of_samples_per_category)

print("   data shape:")
print(training_data.shape)
print(training_label.shape)

print("Start training on data.")

model1 = createModel()
model1.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(training_data, training_label,
           batch_size=batch_size, epochs=epochs, verbose=2)

print("Evaluate model.")

input_image = cv2.imread('pictures/' + args.testimage)
output_image = cv2.imread('pictures/out/' + args.testimage)
training_data, training_label = get_data(input_image,
                                         output_image,
                                         num_of_samples_per_category)

evaluate = model1.evaluate(training_data, training_label,
                           batch_size=batch_size, verbose=2)

print("   evaluate:")
print(evaluate)
