import torch
from transform import TraceTransform
from torchvision import datasets, models, transforms


tx = transforms.Compose([transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,)),
  transforms.Resize((64,64)),
  TraceTransform()])

device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
batch_size_train = 64#16
batch_size_test = 1000

mnist = datasets.MNIST('/files/', train=True, download=True, transform=tx),
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('/files/', train=True, download=True, transform=tx),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('/files/', train=False, download=True, transform=tx),
  batch_size=batch_size_test, shuffle=True)

n = len(mnist[0])
data = []
for i in range(n):
    if i % 10 == 0:
        print(i)
    data.append(mnist[0][i])
torch.save(data, 'train.pth')