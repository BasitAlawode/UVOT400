import numpy as np
import torch


def global_stretching2(img_L, height, width):
    length = height * width
    R_rray = (np.copy(img_L)).flatten()
    R_rray.sort()
    I_min = int(R_rray[int(length / 100)])
    I_max = int(R_rray[-int(length / 100)])
    array_Global_histogram_stretching_L = np.zeros((height, width))
    # print(array_Global_histogram_stretching_L.shape)
    # print(img_L.shape)
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 100
            else:
                p_out = int((img_L[i][j] - I_min) * ((100) / (I_max - I_min)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)


def global_stretching(img_L, height, width):
    length = height * width
    R_rray = (np.copy(img_L)).flatten()
    R_rray.sort()
    I_min = int(R_rray[int(length / 100)])
    I_max = int(R_rray[-int(length / 100)])
    I_min_tensor = torch.tensor(I_min, device='cuda:0')
    I_max_tensor = torch.tensor(I_max, device='cuda:0')
    img_L_tensor = torch.tensor(img_L, device='cuda:0')
    # print(I_min, I_max)
    img_L_tensor[img_L < I_min] = 0
    img_L_tensor[img_L > I_max] = 100
    index = (I_min < img_L) & (img_L < I_max)

    img_L_tensor[index] = (img_L_tensor[index] - I_min_tensor) * (100 / (I_max_tensor - I_min_tensor))
    img_L = img_L_tensor.cpu()
    img_L = img_L.numpy()
    return img_L
