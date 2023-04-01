import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps


def rotates(img, angle_count=64, size = 0):
    if (size > 0):
        img = img.resize((size, size))
        angle_count = size
    
    w, h = img.size
    rad = 2 * math.pi / angle_count
    sinogram = torch.zeros(angle_count, h, w)

    for i in range(angle_count):
        sinogram[i] = F.to_tensor(img.rotate(np.rad2deg(rad * i), Image.BILINEAR))

    return sinogram

# f1
def radon(sinogram, dim = 2):
    return torch.sum(sinogram, dim)


# f2
def maximum(sinogram, dim = 2):
    scores,_ = torch.max(torch.abs(sinogram), dim)
    return scores

# f3
def prime(sinogram, dim = 2):
    n = sinogram.cpu().numpy()
    diff = torch.from_numpy(np.diff(n, axis=dim))
    return torch.sum(torch.abs(diff), dim).to(sinogram.device)

# f4
def prime_double(sinogram, dim = 2):
    n = sinogram.cpu().numpy()
    diff = torch.from_numpy(np.diff(np.diff(n, axis=dim), axis=dim))
    return torch.sum(torch.abs(diff), dim=dim).to(sinogram.device)

