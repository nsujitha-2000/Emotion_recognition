import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2

from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing import image
import scipy

dataset = pd.read_csv('F:/PyCharm Programs/input/fer2013.csv')

image_data = dataset['pixels']
label_data = dataset['emotion']

sns.set_theme(style="darkgrid")
ax = sns.countplot(x="emotion", data=dataset)

oversampler = RandomOverSampler(sampling_strategy='auto')

image_data, label_data = oversampler.fit_resample(image_data.values.reshape(-1, 1), label_data)
print(image_data.shape, " ", label_data.shape)

label_data.value_counts()

image_data = pd.Series(image_data.flatten())
image_data

image_data = np.array(list(map(str.split, image_data)), np.float32)
image_data /= 255
image_data[:10]

image_data = image_data.reshape(-1, 48, 48, 1)
image_data.shape

label_data = np.array(label_data)
label_data.shape

image_train, image_test, label_train, label_test = train_test_split(image_data, label_data, test_size=0.1, random_state=45)

model = Sequential([
    Input((48, 48, 1)),
    Conv2D(32, (3, 3), strides=(1, 1), padding='valid'),
    BatchNormalization(axis=3),
    Activation('relu'),
    Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
    BatchNormalization(axis=3),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), strides=(1, 1), padding='valid'),
    BatchNormalization(axis=3),
    Activation('relu'),
    Conv2D(128, (3, 3), strides=(1, 1), padding='same'),
    BatchNormalization(axis=3),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), strides=(1, 1), padding='valid'),
    BatchNormalization(axis=3),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(200, activation='relu'),
    Dropout(0.6),
    Dense(7, activation='softmax')
])
model.summary()

# tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_dtype=True)

adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

label_train = np_utils.to_categorical(label_train, 7)

label_test = np_utils.to_categorical(label_test, 7)

history = model.fit(image_train, label_train, epochs=30, validation_data=(image_test, label_test))

print("Accuracy of our model on validation dataset : ", model.evaluate(image_test, label_test)[1] * 100, "%")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

label_predicted = model.predict(image_test)
label_result = []

for k in label_predicted:
    label_result.append(np.argmax(k))
label_result[:10]

label_actual = []

for k in label_test:
    label_actual.append(np.argmax(k))
label_actual[:10]

from sklearn.metrics import classification_report

print(classification_report(label_actual, label_result))

import seaborn as sn

cm = tf.math.confusion_matrix(labels=label_actual, predictions=label_result)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save("model.h5")
fer_json = model.to_json()
with open("pred_model.json", "w") as json_file:
    json_file.write(fer_json)
model.save("model.h5")
