Я использовал python3, тестировал скрипты на:

kali_linux(VERSION = "2020.1", версия ядра = 5.4.0-kali4-amd64)

ubuntu(VERSION = "18.04.3 LTS (Bionic Beaver)", версия ядра = 5.3.0-28-generic)

# Как происходит процесс?:

- переходим по адресу: https://sudoku.com/ru

Т.К. распознование цифр с помощью нейронной сети я сделал именно для этого сайта, для других - работать не будет
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/1.png)

- нажимаем Shift + Print Screen(linux) и выделяем квадрат с судокой, должно получиться следующее:
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/2.png)

- копируем наше изображение с судокой в папку data по следующей директории .../sudoky_solver/data. Внимательно: Путь до папки data не должен содержать русских букв!
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/3.png)

