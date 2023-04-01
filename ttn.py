import torch
import torch.nn as nn
import torchvision.transforms as TV
import functional as F
from torch.autograd import Function

# TODO middle layer trace
# uses for paralell also
class TraceLineMiddle(Function):
    @staticmethod
    def forward(ctx, images, weight, bias):
        #N, F, R, T = sinogram.shape
        N = images.size(0)
        F = 2
        R = images.size(2)
        T = 4
        rim = torch.stack([torch.stack([img, img.t().flip(1), img.flip(1), img.flip(1).t()]) for img in images])
        v = rim.view(N, R * T, R)
        x = v.sum(2) # f(x) on n, 2 * 2, 2 -> n, 4  
        max,_ = torch.max(torch.abs(v), 2)
        sinogram = torch.cat([x,max],1)
        o = sinogram.permute(0, 1, 3, 2).contiguous().view(N * F * T, R)
        h = o * weight.t()
        h += bias
        h = h.view(N, F, T, R)
        ctx.save_for_backward(o, weight, bias)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        N, F, R, T = grad_output.shape
        grad_bias = grad_output.sum(2)#keepdim=True)
        grad_y = grad_output.permute(0, 1, 3, 2).contiguous().view(N * F * R, T)
        grad_x = grad_y.t() * weight
        grad_weight = (x * grad_y).t().contiguous().sum(1).view(R, 1)
        return grad_x, grad_weight, grad_bias


class TraceLineMiddle2d(nn.Module):
    def __init__(self, R):
        super(TraceLineMiddle2d, self).__init__()
        self.filter = nn.Parameter(torch.randn(R, 1))
        self.bias = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        return TraceLine.apply(x, self.filter, self.bias)

# arch
# img -> Linear -> ReLU -> Linear
# img -> Trace Layer -> ReLU -> Linear
class TraceLine(Function):
    @staticmethod
    def forward(ctx, sinogram, weight, bias):
        N, F, R, T = sinogram.shape
        #tf = sinogram.view(N, T * R, R)
        #fx = [F.radon, F.maximum]#, F.prime, F.prime_double]
        #F = len(fx)
        #o = torch.cat([f(tf) for f in fx], 1)
        o = sinogram.permute(0, 1, 3, 2).contiguous().view(N * F * T, R)
        h = o * weight.t()
        h += bias
        h = h.view(N, F, T, R)
        ctx.save_for_backward(o, weight, bias)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        N, F, R, T = grad_output.shape
        grad_bias = grad_output.sum(2)#keepdim=True)
        grad_y = grad_output.permute(0, 1, 3, 2).contiguous().view(N * F * R, T)
        grad_x = grad_y.t() * weight
        grad_weight = (x * grad_y).t().contiguous().sum(1).view(R, 1)
        return grad_x, grad_weight, grad_bias


class TraceAngle(Function):
    @staticmethod
    def forward(ctx, sinogram, weight, bias):
        N, F, R, T = sinogram.shape
        # size become N, T * R => N, T * R * n
        o = sinogram.view(N * F * R, T)
        h = o * weight.t()
        h += bias
        h = h.view(N, F, R, T)
        ctx.save_for_backward(o, weight, bias)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        N, n, R, T = grad_output.shape
        grad_bias = grad_output.sum(2)#keepdim=True)
        grad_y = grad_output.view(N * n * R, T)
        grad_x = grad_y.t() * weight
        grad_weight = (x * grad_y).t().contiguous().sum(1).view(R, 1)
        return grad_x, grad_weight, grad_bias

class TraceLineAngle(Function):
    @staticmethod
    def forward(ctx, sinogram, weight, bias):
        N, F, R, T = sinogram.shape
        # size become N, T * R => N, T * R * n
        o = sinogram * weight
        h = o + bias
        ctx.save_for_backward(o, weight, bias)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        N, n, R, T = grad_output.shape
        grad_bias = grad_output.sum(2)#keepdim=True)
        grad_x = grad_output * weight.t()
        grad_weight = x.view(N * n * R, T).t().mm(grad_output.view(N * n * R, T))
        return grad_x, grad_weight, grad_bias

class TraceLine2d(nn.Module):
    def __init__(self, R):
        super(TraceLine2d, self).__init__()
        self.filter = nn.Parameter(torch.randn(R, 1))
        self.bias = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        return TraceLine.apply(x, self.filter, self.bias)

class TraceAngle2d(nn.Module):
    def __init__(self, T):
        super(TraceAngle2d, self).__init__()
        self.filter = nn.Parameter(torch.randn(T, 1))
        self.bias = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        return TraceAngle.apply(x, self.filter, self.bias)

class TraceLineAngle2d(nn.Module):
    def __init__(self, R, T):
        super(TraceLineAngle2d, self).__init__()
        self.filter = nn.Parameter(torch.randn(R, T))
        self.bias = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        return TraceLineAngle.apply(x, self.filter, self.bias)



class TraceLayer(nn.Module):
    def __init__(self):
        super(TraceLayer, self).__init__()
        R = 64
        T = 64
        self.features = nn.Sequential(
            TraceLine2d(R),
            nn.ReLU(),
            nn.MaxPool2d(1, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 512),            
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 217)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# net = TraceLayer()
# # images, subwords, letters, lengths = torch.load('data/jawi.pt')
# images, labels, _, _ = torch.load("data/subword_rotates.pt")
# samples = images[0][0].unsqueeze(0)
# out = net(samples)
# print(out.shape)
# # out.backward(torch.randn(1, 1))
# # print(out)
