import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq


def correct_photo(path, filename):
    image = cv2.imread(os.path.join(path, filename))

    pallete, image = change_colors_on_photo(image)

    image = remove_noise(image)

    img_pallet = create_pallete(pallete)

    return write_files(path, filename, img_pallet, image)


def remove_noise(image):
    return image


def change_colors_on_photo(image):
    arr = np.float32(image)
    pixels = arr.reshape((-1, 3))

    n_colors = 12
    max_iter = 20   # Остановка алгоритма после n колличества прохождений
    epsilon = 0.5   # Точность алгоритма
    # Флаг для указания количества раз, когда алгоритм выполняется с использованием различных начальных меток
    n_iteration = 20
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, n_iteration, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(image.shape)

    return palette, quantized


def create_pallete(palette):
    # Формирование картинки с палитрой изменённой фотографии
    n_colors = len(palette)
    img_pallet = np.array([[[0, 0, 0] for i in range(
        n_colors * 50)] for j in range(n_colors * 10)])
    for i in range(n_colors * 10):
        for j in range(n_colors * 50):
            img_pallet[i][j] = palette[j // 50]

    return img_pallet


def write_files(path, filename, img_pallet, quantized):
    # Запись палитры
    filename_pallet = str(filename.split('.')[0] +
                          "_pallet." + filename.split('.')[1])
    cv2.imwrite(os.path.join(path, filename_pallet), img_pallet)

    # Запись изменённой фотографии
    filename = str(filename.split('.')[0] +
                   "_change." + filename.split('.')[1])
    cv2.imwrite(os.path.join(path, filename), quantized)

    return [filename, filename_pallet]
