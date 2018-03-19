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
resize_factor = 0.25
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

# load images
image = cv2.imread('pictures/' + args.image)
input_image = cv2.resize(image, (0,0), fx=resize_factor, fy=resize_factor)

image = cv2.imread('pictures/out/' + args.image)
output_image = cv2.resize(image, (0,0), fx=resize_factor, fy=resize_factor)

image = cv2.imread('pictures/' + args.testimage)
input_testimage = cv2.resize(image, (0,0), fx=resize_factor, fy=resize_factor)

image = cv2.imread('pictures/out/' + args.testimage)
output_testimage = cv2.resize(image, (0,0), fx=resize_factor, fy=resize_factor)

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
        training_label.append(0.00)

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


print("\nCreate some training data with labels.")
print("--------------------------------------")

training_data, training_label = get_data(input_image,
                                         output_image,
                                         num_of_samples_per_category)

print("   data shape:")
print(training_data.shape)
print(training_label.shape)

print("\nStart training on data.")
print("--------------------------------------")

model1 = createModel()
model1.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(training_data, training_label,
           batch_size=batch_size, epochs=epochs, verbose=2)

print("\nEvaluate model.")
print("--------------------------------------")

test_data, test_label = get_data(input_testimage,
                                 output_testimage,
                                 num_of_samples_per_category)

evaluate = model1.evaluate(test_data, test_label, batch_size=batch_size,
                           verbose=2)

print("   evaluate:")
print(evaluate)

print("\nPredict model.")
print("--------------------------------------")

data = []
width, length, _ = input_image.shape
for y in range(length-34):
    for x in range(width-34):
        input_matrix = input_image[x:(x+32),y:(y+32)]
        data.append(input_matrix)
data = np.array(data)

print("    data_shape:")
print(data.shape)

data_length = data.shape[0]
output = []
step = batch_size * 5
i = 0

while i < data_length:
    print("proccessing %i:%i of %i samples" % (i, i+step, data_length))
    p = model1.predict(data[i:i+step],
                       batch_size=batch_size, verbose=2)
    p = p.reshape((-1, len(p)))
    output.extend(p[0])

    i += step

output = np.array(output).reshape((width-34, -1))

img = cv2.merge((output, output, output))
plt.imshow(img, 'gray')
plt.show()
