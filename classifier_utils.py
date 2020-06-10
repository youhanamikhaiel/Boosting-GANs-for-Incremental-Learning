import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from classifier.resnetw import resnet20
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle
from sklearn.metrics import average_precision_score

#initialize training paramteres
train_batch_size = 125
test_batch_size = 400
verbose = 0
epochs = 132
pos_weight = 18
alpha = 0.9
threshold = 1.4
weight_decay = 0.0001
lr0 = 0.1
num_classes = 10


def ctime():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return(time.time())


def getScoresLabels(net, loader, device):
    # Not memory efficient
    lscores = []
    llabels = []
    net.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            lscores.append(outputs)
            llabels.append(F.one_hot(labels, torch.tensor(outputs.size())[1].item()))
    return torch.cat(lscores), torch.cat(llabels)


def getRates(scores, labels, threshold = 0):
    num_classes = torch.tensor(scores.size())[1].item()
    rates = torch.zeros(4, num_classes, dtype=torch.int).to(device)
    predicted = (scores > threshold).int()
    # true negatives, false negative, false positive, true positive
    rates[0] = torch.sum(((predicted == 0) & (labels == 0)).int(), dim=0)
    rates[1] = torch.sum(((predicted == 0) & (labels == 1)).int(), dim=0)
    rates[2] = torch.sum(((predicted == 1) & (labels == 0)).int(), dim=0)
    rates[3] = torch.sum(((predicted == 1) & (labels == 1)).int(), dim=0)
    return rates

  
def accuracy(scores, labels):
    num_classes = torch.tensor(scores.size())[1].item()
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    predicted = torch.argmax(scores, 1)
    truelabel = torch.argmax(labels, 1)
    c = (predicted == truelabel).squeeze()
    for i in range(list(scores.shape)[0]):
        label = truelabel[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
    return class_correct, class_total



def evaluate(net, loader, device, set = 'Test'):
    t0 = ctime()
    scores, labels = getScoresLabels(net, loader, device)
    # class_correct, class_total = accuracy(scores, labels)
    class_correct, class_total = accuracy(scores, labels)
    print('%s time: %3.2fs' % (set, ctime()-t0), end = "  ")
    print('Accuracy: %2.2f %%' % (100 * sum(class_correct) / sum(class_total)), end = "  ")
    labels = labels.to("cpu").numpy()
    scores = scores.to("cpu").numpy()
    map = torch.tensor(average_precision_score(labels, scores, average=None)).float().mean().item()
    print('%s MAP: %2.2f %%' % (set, 100 * map))



def train(epoch, net, trainloader, trainset, device, optimizer, scheduler, criterionMC, criterionML, alpha):
    print('Epoch: %3d' % epoch, end = "  ")
    running_loss = 0.0
    t0 = ctime()
    net.train()
    iter_num=0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = torch.reshape(labels.long(),(-1,))
        optimizer.zero_grad()
        outputs = net(inputs)
        if (alpha <= 0): loss = criterionMC(outputs, labels)
        elif (alpha >= 1): loss = criterionML(outputs, F.one_hot(labels,num_classes).type_as(outputs))
        else: loss = (1-alpha)*criterionMC(outputs, labels) + \
            alpha*criterionML(outputs, F.one_hot(labels,num_classes).type_as(outputs))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iter_num +=1
        if (iter_num%400) == 0:
          scheduler.step()
      print('Train time: %3.2fs' % (ctime()-t0), end = "  ")
      print('Train loss: %5.4f' % (running_loss*train_batch_size/len(trainset)), end = "  ")
