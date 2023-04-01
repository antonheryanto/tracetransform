from collections import OrderedDict
import time
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import torch.optim as optim


device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
model = models.resnet18() #load resnet18 model
num_features = model.fc.in_features #extract fc layers features
model.fc = nn.Linear(num_features, 10) #(num_of_class == 2)
model = model.to(device)

dataroot = "D:\\TTN-UKM\\FMD\\image\\"
#MNISTDataSetFolder = "D:\\TTN-UKM\\MNIST\\"
batch_size = 16

#MNIST
batch_size_train = 64
batch_size_test = 1000

image_size = (256, 192) #original image  512, 384

tensor_data = transforms.Compose([
    transforms.Resize(image_size),   #must same as here
    #transforms.CenterCrop(image_size),   #-- Take CenterCrop
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

dataset = datasets.ImageFolder(dataroot, tensor_data)

amt_data = len(dataset)
trainamt = int(amt_data * 0.80)

train_dataset, test_dataset = random_split(dataset, [trainamt, amt_data - trainamt])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

def training():
    

    #model = models.resnet18() #load resnet18 model
    
    lossFn = nn.CrossEntropyLoss() #(set loss function)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 200
    for epoch in range(epochs):
        for index, (actual, labels) in enumerate(train_dataloader):
            actual = actual.to(device)
            label = labels.to(device)
            optimizer.zero_grad() # optimizer mengosongkan gradient setiap memulai iterasi // deep learning pakai adam optimizer , bukan scd (stochastic gradient decent)
            #pred = unet(actual)
            score = model.train()(actual) # forward gradient
            _, pred = torch.max(score, 1)
            #print(pred, label)
            loss = lossFn(score, label)
        
            loss.backward() # adjust gradient dalam network (gradient ---> weighted)
            optimizer.step()
        print("epoch: [{}/{}], loss {:.4f}".format(epoch + 1, epochs, loss.data))

def test():

    model.eval()
    with torch.no_grad():
        test_corrects = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data).item()
            test_acc = test_corrects / len(test_dataset) * 100.
    return test_acc

def trainingResNet():
    
    learning_rate = 1e-2
    step = 0
    step_loss = 100
    step_check = 1000
    decay_steps = 10000
    decay_rate = 0.9
    batch_size = 50
    best_accuracy = 0.0
    patience = 10
    duration = 0.0
    #model = models.resnet18() #load resnet18 model
    
    lossFn = nn.CrossEntropyLoss() #(set loss function)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #epochs = 200
    #for epoch in range(epochs):
    while True:
        for index, (actual, labels) in enumerate(train_dataloader):
            start_time = time.time()
            actual = actual.to(device)
            label = labels.to(device)
            optimizer.zero_grad() # optimizer mengosongkan gradient setiap memulai iterasi // deep learning pakai adam optimizer , bukan scd (stochastic gradient decent)
            #pred = unet(actual)
            score = model.train()(actual) # forward gradient
            _, pred = torch.max(score, 1)
            #print(pred, label)
            loss = lossFn(score, label)
        
            loss.backward() # adjust gradient dalam network (gradient ---> weighted)
            optimizer.step()
        
            step += 1
            duration += time.time() - start_time

            if step % step_loss == 0:
                examples_per_sec = batch_size * step_loss / duration
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
                
                #torch.save(
                #    model.state_dict(),
                #    model_file
                #)
                evaluate(model, device)
            else:
                patience -= 1

            print('=> patience = %d' % patience)
            if accuracy >= 100 or patience == 0:
                return



def evaluate(model, device):
    model.eval()
    total = len(test_dataset)
    with torch.no_grad():
        test_corrects = 0
        for inputs, labels in test_dataloader:
            total += len(test_dataset)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data).item()
    return test_corrects, total

if __name__ == '__main__':
    trainingResNet()
    #print(test())

