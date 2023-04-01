import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate

class TraceTransform(nn.Module):
    """Convert image to sinogram using trace transform
       Rotate image and calculate function
    """
    def __init__(self, line_count = 64, angle_count = 64, num_function=4):
        super().__init__()
        self.line_count = line_count
        self.angle_count = angle_count
        self.num_function = num_function
        self.F = [self.radon, self.maximum, self.prime, self.prime_double]

    def forward(self, imgs: torch.Tensor):
        """
        Args:
            img (Tensor): Image to be converted to grayscale.
        Returns:
            Tensor: Grayscaled image.
        """
        c,h,w = imgs.size()
        angle_step = 180 / self.angle_count
        sinogram = torch.zeros([self.num_function, self.angle_count, self.line_count])
        imgr = torch.zeros([self.angle_count, h, w])
        for r in range(self.angle_count):
            imgr[r] = rotate(imgs, r * angle_step)[0]
            for j, F in enumerate(self.F):
                sinogram[j] = F(imgr)
        return sinogram

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_function={self.num_function})"

    @staticmethod # f1
    def radon(img: torch.Tensor, dim = 2):
        return torch.sum(img, dim)

    @staticmethod # f2
    def maximum(img: torch.Tensor, dim = 2) -> torch.Tensor:
        scores,_ = torch.max(torch.abs(img), dim)
        return scores

    @staticmethod # f3
    def prime(img: torch.Tensor, dim = 2):
        return torch.sum(torch.abs(torch.diff(img, dim=dim)), dim)

    @staticmethod # f4
    def prime_double(img: torch.Tensor, dim = 2):
        return torch.sum(torch.abs(torch.diff(img, n=2, dim=dim)), dim)