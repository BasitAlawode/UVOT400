import os
import numpy as np
import cv2
import xlwt
import datetime

from .color_equalisation import RGB_equalisation
from .global_histogram_stretching import stretching
from .hsvStretching import HSVStretching
from .sceneRadiance import sceneRadianceRGB
import time


def UCM(image):
    shape = image.shape
    image = cv2.resize(image, (int(0.25 * shape[1]), int(0.25 * shape[0])))
    sceneRadiance = RGB_equalisation(image)
    sceneRadiance = stretching(sceneRadiance)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    sceneRadiance = cv2.resize(sceneRadiance, (int(shape[1]), int(shape[0])))
    return sceneRadiance

# if __name__ == '__main__':
#
#     imgfolder = r"D:\pythonProject\Uhead\Uhead5\class\test\img\\"
#     img_dirs = os.listdir(imgfolder)
#     saveFolder = r'D:\pythonProject\Uhead\11\\'
#     j = 1
#     for i in img_dirs:
#         img_dir = imgfolder + i
#         # print(img_dir)
#         image = cv2.imread(img_dir)
#         shape = image.shape
#         # print(shape[0])
#         image = cv2.resize(image, (int(0.25 * shape[1]), int(0.25 * shape[0])))
#         # image = cv2.resize(image, (128, 128))
#         # print(image)q8z   90[p-
#         save_dir = saveFolder + str(j) + '.jpg'
#
#         height = len(image)
#         width = len(image[0])
#         start = time.time()
#         sceneRadiance = RGB_equalisation(image)
#         sceneRadiance = stretching(sceneRadiance)
#
#         sceneRadiance = HSVStretching(sceneRadiance)
#         sceneRadiance = sceneRadianceRGB(sceneRadiance)
#         end = time.time()
#
#         sceneRadiance = cv2.resize(sceneRadiance, (int(shape[1]), int(shape[0])))
#         cv2.imwrite(save_dir, sceneRadiance)
#
#         print(j, end - start, '---------------------------------------------------------------------------------------')
#         j = j + 1
