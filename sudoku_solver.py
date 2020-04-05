#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import operator
import imutils
import numpy as np
from glob import glob
from uuid import uuid4
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
try:
    from natsort import natsorted
except:
    from natsort import natsort
from progress.bar import IncrementalBar
from skimage.io import imread as sk_imread
from tensorflow.keras.models import load_model
from skimage.transform import resize as sk_resize
##################################################################################################################################################################################################################
def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    return proc
##################################################################################################################################################################################################################
def find_corners_of_largest_polygon(processed):
    contours, h = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
##################################################################################################################################################################################################################
def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))
##################################################################################################################################################################################################################
def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))
##################################################################################################################################################################################################################
def infer_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares
##################################################################################################################################################################################################################
def cut_from_rect(img, rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
##################################################################################################################################################################################################################
def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]
    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))
##################################################################################################################################################################################################################
def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    img = inp_img.copy()
    height, width = img.shape[:2]
    max_area = 0
    seed_point = (None, None)
    if scan_tl is None:
        scan_tl = [0, 0]
    if scan_br is None:
        scan_br = [width, height]

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)


            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point
##################################################################################################################################################################################################################
def extract_digit(img, rect, size):
    digit = cut_from_rect(img, rect)
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)
##################################################################################################################################################################################################################
def get_digits(img, squares, size):
    digits = []
    img = pre_process_image(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits
##################################################################################################################################################################################################################
def show_image(img):
    name = str(uuid4()) + ".png"
    path_obr_pict = abs_path_folder + "/" + name
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    cv2.imwrite(str(path_obr_pict), img)
##################################################################################################################################################################################################################
def show_digits(digits, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    show_image(np.concatenate(rows))
##################################################################################################################################################################################################################
def display_rects(in_img, rects, colour = 10):
    img = in_img.copy()
    for rect in rects:
        img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    show_image(img)
    return img
##################################################################################################################################################################################################################
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    show_image(img)
    return img
##################################################################################################################################################################################################################
def get_find_contours(image):
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((0, 100, 100), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max )

    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(contours)
    displayCnt = None
    lists = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            lists.append(displayCnt)
    return lists
##################################################################################################################################################################################################################
def get_recognition_of_numbers_in_each_square(countor, image):
    output = four_point_transform(image, countor.reshape(4, 2))
    # cv2.imshow("output", output)
    # cv2.waitKey()
    return output
##################################################################################################################################################################################################################
def save_file(output, name):
    base_folder = "./"
    abs_path_folder = os.path.abspath(base_folder) + "/data"
    path_obr_pict = abs_path_folder + "/" + str(name) + ".png"
    if not os.path.exists(path_obr_pict):
        cv2.imwrite(str(path_obr_pict), output)
##################################################################################################################################################################################################################
def read_image(f, img_width = 150, img_height = 150, channels = 3, mode = 'reflect', anti_aliasing = True):
    return sk_resize(sk_imread(f), output_shape = (img_width, img_height, channels), mode = mode, anti_aliasing = anti_aliasing)
#####################################################################################################################################################################################################################
def read_images(f, img_width = 150, img_height = 150, channels = 3):
    return np.array([read_image(f, img_width, img_height, channels)])
#####################################################################################################################################################################################################################
def get_read_all_png(f):
    size = 150
    img = read_images(str(f), size, size, 3) # читаем картинку и переводим ее в понятный для ПК вид(т.е. цифры)
    model = load_model("digit_recognition.hdf5") # загружаем нейросеть с помощью tensorflow
    predictions = model.predict(img)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    predictions = np.argmax(predictions)
    if len(lists_all_numbers) == 9:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 19:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 29:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 39:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 49:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 59:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 69:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 79:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    elif len(lists_all_numbers) == 89:
        lists_all_numbers.append("\n")
        lists_all_numbers.append(str(classes[predictions]))
    else:
        lists_all_numbers.append(str(classes[predictions]))
#####################################################################################################################################################################################################################
def save_numb_in_file(lists_all_numbers):
    for numb_regocnition in lists_all_numbers:
        with open("numb_regocnition.txt", "a") as f:
            f.write(str(numb_regocnition))
#####################################################################################################################################################################################################################
def norm(a):
    return (a / 9) - .5
#####################################################################################################################################################################################################################
def denorm(a):
    return (a + .5) * 9
##################################################################################################################################################################################################################
def inference_sudoku(sample):
    model = load_model("sudoku_solver.hdf5")
    feat = sample
    while(1):
        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis = 1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis = 1).reshape((9, 9)), 2)

        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if(mask.sum() == 0):
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)

    return pred
#####################################################################################################################################################################################################################
def solve_sudoku(game):
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9, 9, 1))
    game = norm(game)
    game = inference_sudoku(game)
    return game
#####################################################################################################################################################################################################################
def read_txt():
    path = "./"
    abs_path_txt = os.path.abspath(path + "numb_regocnition.txt")
    with open(str(abs_path_txt)) as f:
        os.remove(abs_path_txt)
        return f.read()
#####################################################################################################################################################################################################################
def watch_normaliz(game):
    for i in range(len(game)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")
        for j in range(len(game[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end = "")
            if j == 8:
                print(game[i][j])
            else:
                print(str(game[i][j]) + " ", end="")
##################################################################################################################################################################################################################
def main():
    base_folder = "./data"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    global abs_path_folder
    abs_path_folder = os.path.abspath(base_folder)
    if not glob(os.path.join(abs_path_folder, "*.png")):
        print("\n\nНет изображений с разрешением: '.png' в папке: data\n\n")
    else:
        for file in glob(os.path.join(abs_path_folder, "*.png")):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            processed = pre_process_image(img)
            corners = find_corners_of_largest_polygon(processed)
            # display_points(processed, corners)
            cropped = crop_and_warp(img, corners)
            squares = infer_grid(cropped)
            # display_rects(cropped, squares)
            digits = get_digits(cropped, squares, 42)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            show_digits(digits)
            os.remove(file)

            name = 82
            for file in glob(os.path.join(abs_path_folder, "*.png")):
                image = cv2.imread(str(file))
                find_contours = get_find_contours(image)
                os.remove(file)
                for countor in find_contours:
                    output = get_recognition_of_numbers_in_each_square(countor, image)
                    name -= 1
                    save_file(output, name)

            global lists_all_numbers
            lists_all_numbers = [] # Список для распознаных цифр
            lists_png = [] # Список для путей *.png
            try:
                for f in natsorted(glob(os.path.join(abs_path_folder, "*.png"))): # natsorted - сортируем файлы в нормальной последовательности(01.png, 02.png, 03.png ....)
                    lists_png.append(f) # записываем пути файлов *.png в lists_png
            except:
                for f in natsort(glob(os.path.join(abs_path_folder, "*.png"))): # natsorted - сортируем файлы в нормальной последовательности(01.png, 02.png, 03.png ....)
                    lists_png.append(f) # записываем пути файлов *.png в lists_png
            bar = IncrementalBar('\n\nРАСПОЗНОВАНИЕ ЦИФР, ПОДОЖДИТЕ...', max = len(lists_png)) # Подключаем статус_бар
            for f in lists_png:
                bar.next() # шаг статус_бара
                get_read_all_png(f) # передаем функции с н.сетью для распознования цифр
                os.remove(f) # удаляем файлы *.png
            bar.finish()
            save_numb_in_file(lists_all_numbers) # Сохраняем распознаные цифры в файле

            game = read_txt()
            game = solve_sudoku(game)
            watch_normaliz(game)
##################################################################################################################################################################################################################
if __name__ == "__main__":
    main()
