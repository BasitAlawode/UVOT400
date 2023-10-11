import numpy as np
import math
import torch

e = math.e


def global_Stretching_ab2(a, height, width):
    array_Global_histogram_stretching_L = np.zeros((height, width), 'float64')
    for i in range(0, height):
        for j in range(0, width):
            p_out = a[i][j] * (1.3 ** (1 - math.fabs(a[i][j] / 128)))
            array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)


def global_Stretching_ab(a, height, width):
    a_tensor = torch.tensor(a, device='cuda:0')
    a_tensor = a_tensor * (1.3 ** (1 - abs(a_tensor) / 128))
    a_tensor = a_tensor.cpu()
    a = a_tensor.numpy()
    return a
