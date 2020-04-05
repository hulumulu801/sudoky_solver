# В РАЗРАБОТКЕ...

Я использовал python3, тестировал скрипты на:

**kali_linux**(VERSION = "2020.1", версия ядра = 5.4.0-kali4-amd64)

**ubuntu**(VERSION = "18.04.3 LTS (Bionic Beaver)"; версия ядра = 5.3.0-28-generic; процессор - AMD® Ryzen 7 1800x eight-core processor × 16; графика - GeForce GTX 1080 Ti, NVIDIA-SMI 440.64, Driver Version: 440.64, CUDA Version: 10.2)

# Как происходит процесс?:

- переходим по адресу: https://sudoku.com/ru

Т.К. распознование цифр с помощью нейронной сети я сделал именно для этого сайта, для других - работать не будет
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/1.png)

- нажимаем Shift + Print Screen(linux) и выделяем квадрат с судокой, должно получиться следующее:
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/2.png)

- копируем наше изображение с судокой в папку data по следующей директории .../sudoky_solver/data. Внимательно: Путь до папки data не должен содержать русских букв!
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/3.png)

- запускаем скрипт sudoku_solver.py(python3 sudoku_solver.py), он выполняет несколько этапов:

	* этап_1: с помощью cv2 обрежет изображение до квадрата, выделит линии, отрисует цифры. Это сделано для того, чтобы нейросеть лучше воспринимала изображения с цифрами.
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/4.png)

	* этап_2: так же, с помощью cv2, разбиваем картинку на мелкие квадраты, изображений в папке станет - 81, ровно столько, сколько квадратов.
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/5.png)

	* этап_3: далее начнется распознавание картинок (.png) и превращение их в тексотовый файл (numb_regocnition.txt).
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/6.png)
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/7.png)

	* этап_4: сдесь происходит решение судоку и вывод решения в терминал.
![Image alt](https://github.com/hulumulu801/sudoky_solver/blob/master/picts/8.png)

# Описание файлов:

	- оСНОВНЫЕ ФАЙЛЫ:

		data - папка, в которую необходимо закинуть скрин как показано выше

		digit_recognition.hdf5 - веса нейросети, которая распознаёт цифры на изображениях.

		install.py - установочник для linux(kali, ubuntu)
		
		reqairement.txt - зависимости для python3
		
		sudoku_solver.hdf5 - веса нейронки, которая решает судоку. (Т.К. github ограничил загрузку файлов до 25MB, я закину нейронку на google_disk)
		
		sudoku_solver.py - основной скрипт
		
	- ВСПОМОГАТЕЛЬНЫЕ ФАЙЛЫ В ПАПКЕ other(ФАЙЛЫ, КОТОРОЫЕ НЕ УЧАСТВУЮТ В РАСПОЗНОВАНИИ И РЕШЕНИИ СУДОКУ, О НИХ Я РАССКАЖУ НИЖЕ):

		colab_train_pict.py - файл, как я создавал н.сеть digit_recognition.hdf5

		colab_train_solver_keras_tuner.py - файл, как я искал лучшую модель для решения судоку с использованием kerastuner

		colab_train_sudoku_solver.py - самая простая сеть для решения судоку

		data.tar.gz.bz2 - архив с изображениями для обучения н.сети

		script_copy.py - скрипт, который создаст папки test, train, val и переместит в эти папки необходимое кол-во изображений для обучения н.сети

# Как установить?:

Внимание: Путь не должен содержать русских букв!!

- Если kali_linux или ubuntu:

	РЕКОМЕНДУЮ ИСПОЛЬЗОВАТЬ ПЕРЕД ЗАПУСКОМ: virtualenv

	* открываем терминал и вставляем следующее содержимое:

		sudo apt-get update

		sudo apt-get upgrade

		sudo apt-get dist-upgrade

		sudo apt install git

		cd --


		git clone https://github.com/hulumulu801/sudoky_solver.git
		
		cd sudoky_solver/

		python3 install.py
		
- Если что-то пошло не так или другая ОС:

	* скачиваем с github sudoku_solver(с помощью командной строки и git):
	
		git clone https://github.com/hulumulu801/sudoky_solver.git
		
		или архивом:
		
		https://github.com/hulumulu801/sudoky_solver/archive/master.zip
		
	* скачиваем sudoku_solver.hdf5 с моего google_disk:
	
		https://drive.google.com/uc?export=download&id=1sX8qmDlamLpeTc4QMzx3eg7ZziZOFn0L

		распаковывпем его

	* обновляем pip и setuptuls:

		pip3 install --upgrade pip

		pip3 install --upgrade setuptools

	* устанавливаем pip'ы:

		pip3 install -r reqairement.txt

	* далее устанавливаем tensorflow:
	
		Тут внимательно, т.к. не все процессоры поддерживают инструкцию avx. Если ваш процессор не поддерживает инструкцию avx, ищем tf без инструкции avx. Мне помог вот этот релиз(https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl)
