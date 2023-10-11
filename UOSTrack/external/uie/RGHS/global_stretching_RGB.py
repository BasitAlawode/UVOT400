import cv2
import numpy as np
import torch


def stretching2(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel = np.max(img[:, :, k])
        print(Max_channel)
        Min_channel = np.min(img[:, :, k])
        print(Min_channel)
        for i in range(height):
            for j in range(width):
                img[i, j, k] = (img[i, j, k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel) + 0
    # print(img)
    return img


def stretching(img):
    img_tensor = torch.tensor(img, dtype=torch.float32, device='cuda:0')
    img_tensor = img_tensor.permute(2, 0, 1)
    for k in range(0, 3):
        Max_channel = torch.max(img_tensor[k])
        Min_channel = torch.min(img_tensor[k])
        # print(img_tensor[k])
        img_tensor[k] = (img_tensor[k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel) + 0
    img_tensor = img_tensor.permute(1, 2, 0)
    img = img_tensor.cpu().numpy()
    # print(img)
    return img


if __name__ == '__main__':
    img = cv2.imread(r'D:\pythonProject\Uhead\11\\1.jpg')
    print(img.shape)
    img_tensor = torch.tensor(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    print(img_tensor.shape)
    img_tensor2 = img_tensor[0]
    print(img_tensor2.shape)
    Max_channel = torch.max(img_tensor[0])
    print(Max_channel)
    img_tensor = img_tensor.permute(1, 2, 0)
    print(img_tensor.shape)
