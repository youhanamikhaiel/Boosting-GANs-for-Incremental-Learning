from ClassifierLoader import ClassifierDataset, RandomCrop, RandomHorizontalFlip
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
from classifier_utils import ctime, getScoresLabels, getRates, accuracy, evaluate, train



def main():
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

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = transforms.Compose([RandomHorizontalFlip() , RandomCrop(size=32,padding=4), normalize])
    trainset = ClassifierDataset('samples/samples_total.npz',transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=125, shuffle=True, num_workers=8, pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=400, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = resnet20().to(device)
    
    criterionMC = nn.CrossEntropyLoss()
    criterionML = nn.BCEWithLogitsLoss(pos_weight=pos_weight*torch.ones([num_classes]).to(device))
    optimizer = optim.SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2,(3*epochs)//4])
    
    t1 = ctime()
    for epoch in range(epochs):  # loop over the dataset multiple times
        train(epoch, net, trainloader, device, optimizer, scheduler, criterionMC, criterionML, alpha)
        evaluate(net, testloader, device)
        scheduler.step()
    tt = ctime()-t1
    print('Finished Training, total time %4.2fs' % (tt))


    t0 = ctime()
    scores, labels = getScoresLabels(net, testloader, device)
    class_correct, class_total = accuracy(scores, labels)
    print('Test time: %3.2fs' % (ctime()-t0))

    print('Overall accuracy  : %2.2f %%' % (100 * sum(class_correct) / sum(class_total)))
    print()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(num_classes):
        print('Accuracy of %5s : %2.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
  


if __name__ == '__main__':
    main()