import numpy as np
import torch


def cal_equalisation(img, ratio):
    img_tensor = torch.tensor(img, device='cuda:0')
    ratio_tensor = torch.tensor(ratio, device='cuda:0')
    with torch.no_grad():
        Array_tensor = img_tensor * ratio_tensor
    Array = Array_tensor.cpu().numpy()
    # Array = np.uint8(Array)
    Array = np.clip(Array, 0, 255)
    return Array


def RGB_equalisation(img):
    img = np.float32(img)
    avg_RGB = []
    for i in range(3):
        avg = np.mean(img[:, :, i])
        avg_RGB.append(avg)
    # print('avg_RGB',avg_RGB)
    a_r = avg_RGB[0] / avg_RGB[2]
    a_g = avg_RGB[0] / avg_RGB[1]
    ratio = [0, a_g, a_r]
    for i in range(1, 3):
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
    return img
