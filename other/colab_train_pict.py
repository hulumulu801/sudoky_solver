#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from google.colab import drive # подключаем библиотеки с работой google_disk
drive.mount("/content/gdriver", force_remount = True) # монтируем google_disk
########################################################################################################################################################################
!cp -r /content/gdriver/"My Drive"/data_pict.tar . # копируем архив с изображениями
!tar -xvf data_pict.tar # разархивируем
########################################################################################################################################################################
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
########################################################################################################################################################################
# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 5
# Размер мини-выборки
batch_size = 1
# Количество изображений для обучения
nb_train_samples = 8172
# Количество изображений для проверки
nb_validation_samples = 1764
# Количество изображений для тестирования
nb_test_samples = 1755
########################################################################################################################################################################
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
########################################################################################################################################################################
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
datagen = ImageDataGenerator(rescale = 1. / 255)
########################################################################################################################################################################
train_generator = datagen.flow_from_directory(train_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
test_generator = datagen.flow_from_directory(test_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
########################################################################################################################################################################
checkpoint = ModelCheckpoint("digit_recognition.hdf5", monitor = "val_accuracy", save_best_only = True, verbose = 1)
learning_rate_reduction = ReduceLROnPlateau(monitor = "val_accuracy", patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
########################################################################################################################################################################
history = model.fit_generator(train_generator,
                                steps_per_epoch = nb_train_samples // batch_size,
                                epochs = epochs,
                                validation_data = val_generator,
                                validation_steps = nb_validation_samples // batch_size,
                                callbacks = [checkpoint, learning_rate_reduction])
########################################################################################################################################################################
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
########################################################################################################################################################################
plt.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
########################################################################################################################################################################
!cp -r digit_recognition.hdf5 /content/gdriver/"My Drive"/ # Копируем модель себе на Google_disk
