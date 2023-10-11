import numpy as np
import torch


def global_stretching(img_L, height, width):
    I_min = np.min(img_L)
    I_max = np.max(img_L)
    I_min_tensor = torch.tensor(I_min, device='cuda:0')
    I_max_tensor = torch.tensor(I_max, device='cuda:0')
    img_L_tensor = torch.tensor(img_L, device='cuda:0')
    with torch.no_grad():
        img_L_tensor = (img_L_tensor - I_min_tensor) * (1 / (I_max_tensor - I_min_tensor))
    # array_Global_histogram_stretching_L = np.zeros((height, width))
    # for i in range(0, height):
    #     for j in range(0, width):
    #         p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
    #         array_Global_histogram_stretching_L[i][j] = p_out
    img = img_L_tensor.cpu().numpy()
    return img
