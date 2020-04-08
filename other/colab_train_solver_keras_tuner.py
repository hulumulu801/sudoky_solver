#!/usr/bin/env python3
# -*- coding: utf-8
######################################################################################################################
from google.colab import drive # подключаем библиотеки с работой google_disk
drive.mount("/content/gdriver", force_remount = True) # монтируем google_disk
######################################################################################################################
!cp -r /content/gdriver/"My Drive"/sudoku.tar.gz . # копируем архив с sudoku
!tar -xvf sudoku.tar.gz # разархивируем
######################################################################################################################
pip install -U keras-tuner
######################################################################################################################
%tensorflow_version 2.x
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape, Dropout, Input, Activation
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import utils
import numpy as np
import pandas as pd
######################################################################################################################
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
    x = np.array([int(j) for j in i]).reshape((81,1)) - 1
    label.append(x)
label = np.array(label)
del(feat_raw)
del(label_raw)
x_train, x_val, y_train, y_val = train_test_split(feat, label, test_size = 0.2, random_state = 42)
######################################################################################################################
def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values = ['relu', 'sigmoid', 'tanh', 'elu', 'selu'])

    model.add(Conv2D(hp.Int("input_units", min_value = 32, max_value = 1024, step = 32), kernel_size = (3, 3), activation = activation_choice, padding = 'same',  input_shape = (9, 9, 1)))
    model.add(BatchNormalization())

    for j in range(hp.Int("num_layers", 2, 4)):
        model.add(Conv2D(hp.Int("input_units_1", min_value = 32, max_value = 1024, step = 32), kernel_size = (3, 3), activation = activation_choice, padding = 'same'))
        model.add(BatchNormalization())

    model.add(Conv2D(hp.Int("input_units_2", min_value = 64, max_value = 1024, step = 32), kernel_size = (1, 1), activation = activation_choice, padding = 'same'))

    model.add(Flatten())
    model.add(Dense(81 * 9))
    model.add(Dropout(0.5))
    model.add(Reshape((- 1, 9)))
    model.add(Activation('softmax'))

    model.compile(
        optimizer = hp.Choice('optimizer', values = ['adam', 'rmsprop', 'SGD']),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

    return model
######################################################################################################################
tuner = BayesianOptimization(
    build_model,
    objective = 'val_accuracy',
    max_trials = 100,
    directory = 'test_directory'
    )
######################################################################################################################
tuner.search_space_summary()
######################################################################################################################
tuner.search(x_train,                  # Данные для обучения
             y_train,                  # Правильные ответы
             batch_size = 64,           # Размер мини-выборки
             epochs = 20,                # Количество эпох обучения
             validation_split = 0.2,     # Часть данных, которая будет использоваться для проверки
             verbose = 1)
######################################################################################################################
tuner.results_summary()
######################################################################################################################
models = tuner.get_best_models(num_models = 3)
######################################################################################################################
for model in models:
  model.summary()
  model.evaluate(x_val, y_val)
  print()
######################################################################################################################
