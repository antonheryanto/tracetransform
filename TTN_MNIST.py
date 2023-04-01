import torch
import torch.nn as nn
import time
from datetime import datetime
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import torch.optim as optim
import model as M
from transform import TraceTransform

device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
batch_size_train = 64#16
batch_size_test = 1000

tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Resize((64,64)),
    TraceTransform(28,28)
])

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

#dataset = datasets.ImageFolder(dataroot, tensor_data)
#amt_data = len(dataset)
#trainamt = int(amt_data * 0.80)
#train_dataset, test_dataset = random_split(dataset, [trainamt, amt_data - trainamt])

#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

def adjust_learning_rate(optimizer, step, initial_lr, decay_steps, decay_rate):
    lr = initial_lr * ((1 - decay_rate) ** (step // decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def evaluate(model, device):
    model.eval()
    total = 0
    with torch.no_grad():
        test_corrects = 0
        for inputs, labels in test_loader:
            total += labels.shape[0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data).item()
            print(f"{test_corrects}/{total}:{test_corrects/total}")
    return test_corrects, total

def tainingTTN():
    
    #lossFn = nn.CrossEntropyLoss()
    
    learning_rate = 1e-2
    step = 0
    step_loss = 1
    step_check = 100
    decay_steps = 10000
    decay_rate = 0.9
    best_accuracy = 0.0
    patience = 10
    duration = 0.0
    is_multi=False
    model = M.TraceLineNet(28) # model yg di pake
    model.to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=0.0005)

    while True:
        for index, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            images = images.to(device)
            label = labels.to(device)
            score = model.train()(images)
            _, pred = torch.max(score, 1)
            loss = model.loss(score, label) #lossFn(score, label)

            learning_rate = adjust_learning_rate(
                optimizer,
                step=step,
                initial_lr=learning_rate,
                decay_steps=10000,
                decay_rate=decay_rate
            )
            
            optimizer.zero_grad()
            loss.backward() # adjust gradient dalam network (gradient ---> weighted)
            optimizer.step()
            
            step += 1
            duration += time.time() - start_time
            if step % step_loss == 0:
                if duration == 0:
                    duration = 1
                examples_per_sec = batch_size_train * step_loss / duration
                duration = 0.0
                print('=> %s: step %d, loss = %f, learning_rate = %f (%.1f examples/sec)' % (
                    datetime.now(), step, loss.item(), learning_rate, examples_per_sec))

            if step % step_check != 0:
                continue

            print('=> Evaluating texture on validation dataset...')
            correct, total = evaluate(model, device)
            accuracy = correct * 100.0/ total
            print('%d/%d==> accuracy = %f, best accuracy %f' %
                  (correct, total, accuracy, best_accuracy))
            
            if accuracy > best_accuracy:
                patience = patience
                best_accuracy = accuracy
                
                torch.save(model.state_dict(), "model.pth")
                #evaluate(model, device, )
            else:
                patience -= 1

            print('=> patience = %d' % patience)
            if accuracy >= 100 or patience == 0:
                return

if __name__ == '__main__':
    # model = M.TraceLineNet(28)
    # x = torch.rand(1,4, 28, 28)
    # y = model(x)
    # print(y.shape)
    tainingTTN()