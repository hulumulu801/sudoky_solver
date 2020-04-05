#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import sys
import subprocess
from glob import glob
##################################################################################################################################################################################
def install_kali_linux_and_ubuntu():
    path = os.path.abspath("./")
# Ищем русские буквы в директории
    search_russian_letters = re.findall("[а-яА-ЯёЁ]+", str(path))
    for search_russian_letter in search_russian_letters:
        pass
    try:
        if re.search(str(search_russian_letter), str(path)):
            print("\nОшибка: Есть русские буквы в директории!\n")
    except:
        abs_path_pip = path + "/" + "reqairement.txt" # путь до reqairement.txt
# обновляем
        os.system("sudo apt-get update -y")
# устанавливаем wget
        os.system("sudo apt install -y wget")
# устанавливаем bzip2
        os.system("sudo apt-get install -y bzip2")
# скачиваем sudoku_solver.hdf5 с моего google_disk
        os.system("wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1sX8qmDlamLpeTc4QMzx3eg7ZziZOFn0L' -O sudoku_solver.hdf5.bz2")
# если файл скачался, то распаковываем, если нет - печатаем ошибку, что файла нет
        if os.path.exists(path + "/" + "sudoku_solver.hdf5.bz2"):
            os.system("bzip2 -d sudoku_solver.hdf5.bz2") # восстановление оригинальной версии файла из сжатой версии
        else:
            print("Файл sudoku_solver.hdf5.bz2 не скачался")
# обновляем pip и setuptuls
        os.system("pip3 install --upgrade pip")
        os.system("pip3 install --upgrade setuptools")
# устанавливаем pip'ы
        os.system("pip3 install -r" + str(abs_path_pip))
# подготавлеваемся к установке tensorflow, т.к. не все процессоры имеют инструкцию avx
        try:
            command = "lscpu | grep 'Flags:'"
            info_cpu_flags = subprocess.check_output(command, shell = True).strip().decode()
        except:
            command = "lscpu | grep 'Флаги:'"
            info_cpu_flags = subprocess.check_output(command, shell = True).strip().decode()
        results = re.findall(r"avx", str(info_cpu_flags))
        if len(results) == 0:
            print("\n\nВаш процессор не имеет инструкцию 'avx'!\nУстановка tensorflow без инструкций SSE4.1, SSE4.2, AVX, AVX2, FMA, MKL\n\n")
# скачиваем tensorflow без SSE4.1, SSE4.2, AVX, AVX2, FMA, MKL
            os.system("wget https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl")
# скаченный файл tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl нужно переименовать, в зависимости от версии python
            command_python_version = "python3 --version"
            info_python_version = subprocess.check_output(command_python_version, shell = True).strip().decode().split(".")[1]
# если python3.1 - переименовываем в tf_nightly-1.13.1-cp31-cp31m-linux_x86_64.whl
            if int(info_python_version) == 1:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp31", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.2 - переименовываем в tf_nightly-1.13.1-cp32-cp32m-linux_x86_64.whl
            elif int(info_python_version) == 2:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp32", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.3 - переименовываем в tf_nightly-1.13.1-cp33-cp33m-linux_x86_64.whl
            elif int(info_python_version) == 3:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp33", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.4 - переименовываем в tf_nightly-1.13.1-cp34-cp34m-linux_x86_64.whl
            elif int(info_python_version) == 4:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp34", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.5 - переименовываем в tf_nightly-1.13.1-cp35-cp35m-linux_x86_64.whl
            elif int(info_python_version) == 5:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp35", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.6 - переименовываем в tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl
            elif int(info_python_version) == 6:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp36", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.7 - переименовываем в tf_nightly-1.13.1-cp37-cp37m-linux_x86_64.whl
            elif int(info_python_version) == 7:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp37", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.8 - переименовываем в tf_nightly-1.13.1-cp38-cp38m-linux_x86_64.whl
            elif int(info_python_version) == 8:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp38", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# если python3.9 - переименовываем в tf_nightly-1.13.1-cp39-cp39m-linux_x86_64.whl
            elif int(info_python_version) == 9:
                paths_to_tf = glob(os.path.join(str(path), "*.whl")) # узнаем абсолютный путь до файла с расширением .whl
                for path_to_tf in paths_to_tf:
                    rename_file = re.sub(r"cp36", "cp39", str(path_to_tf))
                    os.rename(str(path_to_tf), str(rename_file))
# устанавливаем скаченный файл с расширением .whl
            os.system("pip3 install " + str(rename_file))
# удаляем файл с расширением .whl
            os.remove(str(rename_file))
# если avx имеется, устанавливаем tf с помощью pip
        else:
            os.system("pip3 install --upgrade tensorflow")
##################################################################################################################################################################################
def main():
    version_os = sys.platform
    if version_os == "linux":
        install_kali_linux_and_ubuntu()
##################################################################################################################################################################################
if __name__ == "__main__":
    main()
