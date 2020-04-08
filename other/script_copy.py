#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
################################################################################################################
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for i in range(0, 10):
        os.makedirs(os.path.join(dir_name, str(i)))
################################################################################################################
def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        for p in range(0, 10):
            shutil.copy2(os.path.join(source_dir + "/" + str(p) + "/", str(p) + "." + str(i) + ".png"), os.path.join(dest_dir + "/" + str(p) + "/" + str(p) + "." + str(i) + ".png"))
################################################################################################################
def main():
    os.system("bzip2 -d data.tar.gz.bz2")
    os.system("tar -xvf data.tar.gz")
    data_dir = "./data" # каталог с данными
    train_dir = "./train" # Каталог с данными для обучения
    val_dir = "./val" # Каталог с данными для проверки
    test_dir = "./test" # Каталог с данными для тестирования
    test_data_portion = 0.15 # Часть набора данных для тестирования
    val_data_portion = 0.15 # Часть набора данных для проверки
    nb_images = 1300 # Количество данных в одном классе
    start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
    start_test_data_idx = int(nb_images * (1 - test_data_portion))

    create_directory(train_dir)
    create_directory(val_dir)
    create_directory(test_dir)

    copy_images(1, start_val_data_idx, data_dir, train_dir)
    copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
    copy_images(start_test_data_idx, nb_images, data_dir, test_dir)

    os.system("rm -rf " + data_dir) # удаляем каталог data
    os.system("tar -cvf data_pict.tar test/ train/ val/") # добавим к архиву  data_pict.tar.gz папки test, train, val
    os.system("rm -rf " + train_dir + " " + val_dir + " " + test_dir) # удалим папки test, train, val
    os.system("rm -rf data.tar.gz") # удаляем data.tar.gz
################################################################################################################
if __name__ == "__main__":
    main()
