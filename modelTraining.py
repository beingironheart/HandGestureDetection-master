import cv2
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

# actions = np.array(['bye', 'come together', 'Good Morning', 'hello', 'high five', 'how are you?', 'idle', 'I love you', 'listen up', 'Looser', 'me', 'namaste', 'Not Okay', 'okay', 'peace', 'Please call me..!', 'Rock', 'sorry', 'stop it..!', 'unique', 'wrong', 'you'])

actions = ['bye', 'hello', 'you']

sequence_length = 100

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

img_size = 224
# DATA_PATH = os.path.join('Action_Frame', action)
data = []
for action in actions:
    DATA_PATH = os.path.join('Action_Frame', action)
    # print(DATA_PATH)
    # print(os.listdir(DATA_PATH))
    class_num = actions.index(action)
    # print(class_num)
    for img in os.listdir(DATA_PATH):
        try:
            img_arr = cv2.imread(os.path.join(DATA_PATH, img))[..., ::-1]  # convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
            data.append([resized_arr, class_num])
        except Exception as e:
            print(e)

val = []
for action in actions:
    DATA_PATH = os.path.join('Actions_test', action)
    # print(DATA_PATH)
    # print(os.listdir(DATA_PATH))
    class_num = actions.index(action)
    # print(class_num)
    for img in os.listdir(DATA_PATH):
        try:
            img_arr = cv2.imread(os.path.join(DATA_PATH, img))[..., ::-1]  # convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
            val.append([resized_arr, class_num])
        except Exception as e:
            print(e)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in data:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

# model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# opt = Adam(learning_rate=0.000001)
# model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) , metrics = ['accuracy'])
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

history = model.fit(x_train,y_train,epochs = 100 ,  verbose=1, validation_data = (x_val, y_val))

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(100)
#
# plt.figure(figsize=(15, 15))
# plt.subplot(2, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['bye (Class 0)','hello (Class 1)','you (Class 2)']))