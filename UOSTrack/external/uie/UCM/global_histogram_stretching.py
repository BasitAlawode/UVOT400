import numpy as np
import torch


def histogram_r(r_array, height, width):
    length = height * width
    R_rray = (np.copy(r_array)).flatten()
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])

    I_min_tensor = torch.tensor(I_min, device='cuda:0')
    I_max_tensor = torch.tensor(I_max, device='cuda:0')
    r_array_tensor = torch.tensor(r_array, device='cuda:0')
    # print(I_min, I_max)
    r_array_tensor[r_array < I_min] = I_min
    r_array_tensor[r_array > I_max] = 255
    index = (I_min < r_array) & (r_array < I_max)
    with torch.no_grad():
        r_array_tensor[index] = (r_array_tensor[index] - I_min_tensor) * (
                (255 - I_min) / (I_max_tensor - I_min_tensor)) + I_min_tensor
    r_array = r_array_tensor.cpu().numpy()

    r_array = np.uint8(r_array)
    return r_array


def histogram_g(r_array, height, width):
    length = height * width
    R_rray = (np.copy(r_array)).flatten()
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])

    I_min_tensor = torch.tensor(I_min, device='cuda:0')
    I_max_tensor = torch.tensor(I_max, device='cuda:0')
    r_array_tensor = torch.tensor(r_array, device='cuda:0')
    # print(I_min, I_max)
    r_array_tensor[r_array < I_min] = 0
    r_array_tensor[r_array > I_max] = 255
    index = (I_min < r_array) & (r_array < I_max)
    with torch.no_grad():
        r_array_tensor[index] = (r_array_tensor[index] - I_min_tensor) * (255 / (I_max_tensor - I_min_tensor))
    r_array = r_array_tensor.cpu().numpy()
    r_array = np.uint8(r_array)
    return r_array


def histogram_b(r_array, height, width):
    length = height * width
    R_rray = (np.copy(r_array)).flatten()
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])

    I_min_tensor = torch.tensor(I_min, device='cuda:0')
    I_max_tensor = torch.tensor(I_max, device='cuda:0')
    r_array_tensor = torch.tensor(r_array, device='cuda:0')
    # print(I_min, I_max)
    r_array_tensor[r_array < I_min] = 0
    r_array_tensor[r_array > I_max] = I_max
    index = (I_min < r_array) & (r_array < I_max)
    with torch.no_grad():
        r_array_tensor[index] = (r_array_tensor[index] - I_min_tensor) * (I_max_tensor / (I_max_tensor - I_min_tensor))
    r_array = r_array_tensor.cpu().numpy()
    r_array = np.uint8(r_array)
    return r_array


def stretching(img):
    height = len(img)
    width = len(img[0])
    img[:, :, 2] = histogram_r(img[:, :, 2], height, width)
    img[:, :, 1] = histogram_g(img[:, :, 1], height, width)
    img[:, :, 0] = histogram_b(img[:, :, 0], height, width)
    return img
