
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import cv2
import numpy as np

img_name = 'a1.jpg'
input_image = cv2.imread('pictures/' + img_name)
output_image = cv2.imread('pictures/out/' + img_name)

width,length,ch = input_image.shape

training_data = []
training_label = []

for x in range(width-32):
    for y in range(length-32):
        input_matrix = input_image[x:(x+32),y:(y+32)]
        output_scalar = 1 if output_image[x+16,y+16][0] > 0 else 0
        training_data.append(input_matrix)
        training_label.append(output_scalar)

print len(training_data)
print len(training_label)

def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=training_data[0].shape))
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

model1 = createModel()
batch_size = 256
epochs = 3
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(training_data, training_label, verbose=2, batch_size=batch_size, epochs=epochs)
