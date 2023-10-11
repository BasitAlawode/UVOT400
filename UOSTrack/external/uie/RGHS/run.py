# encoding=utf-8
import os
import numpy as np
import cv2

from .LabStretching import LABStretching
from .color_equalisation import RGB_equalisation
from .global_stretching_RGB import stretching
from .relativeglobalhistogramstretching import RelativeGHstretching


def RGHSUWE(image):
    shape = image.shape
    H, W = shape[0], shape[1]
    image = cv2.resize(image, (int(0.25 * W), int(0.25 * H)))
    image = stretching(image)
    image = LABStretching(image)
    image = cv2.resize(image, (W, H))
    return image

#
# imgfolder = r"D:\pythonProject\Uhead\Uhead5\class\test\img\\"
# img_dirs = os.listdir(imgfolder)
# saveFolder = r'D:\pythonProject\Uhead\11\\'
#
# j = 0
# for i in img_dirs:
#     print(j, '--------------------------------------------------------------------------------------------------------')
#     img_dir = imgfolder + i
#     # print(img_dir)
#     image = cv2.imread(img_dir)
#
#     shapea = image.shape
#     H, W = shapea[0], shapea[1]
#     image = cv2.resize(image, (int(0.25 * W), int(0.25 * H)))
#
#     # print(image)
#     save_dir = saveFolder + str(j) + '.jpg'
#     shape = image.shape
#     # print(shape)
#     height = len(image)
#     width = len(image[0])
#
#     sceneRadiance = image
#
#     sceneRadiance = stretching(sceneRadiance)
#
#     sceneRadiance = LABStretching(sceneRadiance)
#
#     image1 = cv2.resize(sceneRadiance, (W, H))
#     cv2.imwrite(save_dir, image1)
#
#     j = j + 1
