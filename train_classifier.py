from ClassifierLoader import ClassifierDataset, RandomCrop, RandomHorizontalFlip
from RealDataLoader import GANDataset
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
import dist_utils
import params
import math



def main(config):
    
    
    """
    Computing minimum distance between real data samples and the generated data distribution
    """
        
    global resnet_real_feats, dist
    dist = [torch.Tensor([]) for _ in range(config['n_classes'])]
    
    #load real data for distance computation
    b_size = 50000
    normalize_real = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_real = transforms.Compose([normalize_real])
    trainset_real = GANDataset( 'samples/real_data/CIFAR10_training.npz', 'BigGANOriginal/data_weights/CIFAR10_weights.npz', transform=transform_real)
    trainloader_real = torch.utils.data.DataLoader(trainset_real, batch_size=b_size, shuffle=False, num_workers=8, pin_memory=True)
    
    resnet_real_feats, indices = dist_utils.get_real_feats(trainloader_real,'resnet20')
    persample_weights = dist_utils.get_sample_weights(resnet_real_feats,indices,config)
    
    del resnet_real_feats, indices
    
    
    """
    Training a resnet20 classifier on generated images
    """
    
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
    
    #import fake training dataloader with proper transormations
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = transforms.Compose([RandomHorizontalFlip() , RandomCrop(size=32,padding=4), normalize])
    trainset = ClassifierDataset('samples/samples_total.npz',transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    #import real testing dataloader 
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    #make sure that cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #import resnet classifier model
    net = resnet20().to(device)
    
    #define optimization and learning hyperparameters
    criterionMC = nn.CrossEntropyLoss()
    criterionML = nn.BCEWithLogitsLoss(pos_weight=pos_weight*torch.ones([num_classes]).to(device))
    optimizer = optim.SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2,(3*epochs)//4])
    
    #calculate number of iteration
    total_iter = epoch*(50000/train_batch_size)
    new_epochs = math.ceil(total_iter/(len(trainset)/train_batch_size))
    
    #main training 
    t1 = ctime()
    for epoch in range(new_epochs):  # loop over the dataset multiple times
        train(epoch, net, trainloader, trainset, device, optimizer, scheduler, criterionMC, criterionML, alpha)
        evaluate(net, testloader, device)
        scheduler.step()
    tt = ctime()-t1
    print('Finished Training, total time %4.2fs' % (tt))

    #print total test accuracy and per-class accuracy
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
    config = params.params
    main(config)

