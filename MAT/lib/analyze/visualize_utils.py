import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(img, box, color, in_sz, out_sz=(750, 600), width=5):

    imh, imw = img.shape[:2]
    sh, sw = out_sz[1] / in_sz[1], out_sz[0] / in_sz[0]

    if imh != out_sz[1] or imw != out_sz[0]:
        img = cv2.resize(img, out_sz, interpolation=cv2.INTER_LINEAR)

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    # draw result
    box = np.array(box)  # [x y x y]
    box[0::2] *= sw
    box[1::2] *= sh
    box = box.astype(int)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=width)

    return np.array(img)
