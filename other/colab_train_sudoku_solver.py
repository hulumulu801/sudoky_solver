#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################################################################################################
from google.colab import drive # подключаем библиотеки с работой google_disk
drive.mount("/content/gdriver", force_remount = True) # монтируем google_disk
########################################################################################################################################################################################################
!cp -r /content/gdriver/"My Drive"/sudoku.tar.gz . # копируем архив с sudoku
!tar -xvf sudoku.tar.gz # разархивируем
########################################################################################################################################################################################################
%tensorflow_version 2.x
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape, Dropout, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
########################################################################################################################################################################################################
file = "sudoku.csv"
data = pd.read_csv(file)
feat_raw = data['quizzes']
label_raw = data['solutions']
feat = []
label = []
for i in feat_raw:
    x = np.array([int(j) for j in i]).reshape((9, 9, 1))
    feat.append(x)
feat = np.array(feat)
feat = feat / 9
feat -= .5
for i in label_raw:
    x = np.array([int(j) for j in i]).reshape((81, 1)) - 1
    label.append(x)
label = np.array(label)
del(feat_raw)
del(label_raw)
x_train, x_val, y_train, y_val = train_test_split(feat, label, test_size = 0.2, random_state = 42)
########################################################################################################################################################################################################
input_shape = (9, 9, 1)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(81 * 9))
model.add(Dropout(0.25))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))
########################################################################################################################################################################################################
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

сheckpoint = ModelCheckpoint("sudoku_solver.hdf5", monitor = "val_accuracy", save_best_only = True, verbose = 1)
learning_rate_reduction = ReduceLROnPlateau(monitor = "val_accuracy", patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
########################################################################################################################################################################################################
batch_size = 10
history = model.fit(x_train, y_train, batch_size = batch_size,
                    epochs = 15,
                    validation_data = (x_val, y_val),
                    verbose = 1,
                    callbacks = [сheckpoint, learning_rate_reduction])
########################################################################################################################################################################################################
plt.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
########################################################################################################################################################################################################
!cp -r sudoku_solver.hdf5 /content/gdriver/"My Drive"/ # Копируем модель себе на Google_disk
