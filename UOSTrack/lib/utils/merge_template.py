import cv2
import numpy as np


def multi_small_template(template_list, attn_list):
    template_list[0] = cv2.resize(template_list[0], (64, 64))
    template_list[1] = cv2.resize(template_list[1], (64, 64))
    template_list[2] = cv2.resize(template_list[2], (64, 64))
    template_list[3] = cv2.resize(template_list[3], (64, 64))

    attn_list[0] = cv2.resize(attn_list[0].astype(np.uint8), (64, 64)).astype(np.bool_)
    attn_list[1] = cv2.resize(attn_list[1].astype(np.uint8), (64, 64)).astype(np.bool_)
    attn_list[2] = cv2.resize(attn_list[2].astype(np.uint8), (64, 64)).astype(np.bool_)
    attn_list[3] = cv2.resize(attn_list[3].astype(np.uint8), (64, 64)).astype(np.bool_)

    template_item1_2 = np.concatenate((template_list[0], template_list[1]), axis=1)
    template_item3_4 = np.concatenate((template_list[2], template_list[3]), axis=1)
    template_item = np.concatenate((template_item1_2, template_item3_4), axis=0)

    attn_item1_2 = np.concatenate((attn_list[0], attn_list[1]), axis=1)
    attn_item3_4 = np.concatenate((attn_list[2], attn_list[3]), axis=1)
    attn_item = np.concatenate((attn_item1_2, attn_item3_4), axis=0)
    return template_item, attn_item
