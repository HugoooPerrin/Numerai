
# coding: utf-8

# Import modules

import pandas as pd
import numpy as np
import sys
import nltk
import pickle
import time

import torch
import torch.nn as nn
import torch.utils.data as utils
import torchwordemb
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import log_loss
from multiprocessing import Pool

# Personal modules
from models import *
from utils import *

time1 = time.time()

# Import data
train_vect = pd.read_csv('../../../Datasets/Numerai/w95/numerai_tournament_data.csv')

Xtrain = pd.read_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/train.csv')
list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_labels = Xtrain[list_classes].values

Xtest = pd.read_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/test.csv')
final_id = Xtest['id']

del Xtrain, Xtest

# Preprocess data for torch
train_comments = train_vect.reshape(train_vect.shape[0],1,train_vect.shape[1])
test_comments = test_vect.reshape(test_vect.shape[0],1,test_vect.shape[1])

del train_vect, test_vect

# Get final predictions
predictions = pd.DataFrame(index=final_id)

for target in list_classes:
    
    print("\nEstimation of {}:".format(target))

    labels_train = train_labels[:,list_classes.index(target)]
    labels_train = labels_train.reshape(labels_train.shape[0],1)

    use_GPU = True

    batch_size = 512
    num_epoch = 4

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_comments), 
                                                   torch.FloatTensor(labels_train))


    test_dataset = torch.FloatTensor(test_comments)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers = 8)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, 
                                               num_workers = 8)

    net = Inception()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.000015, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

    train(num_epoch, net, train_loader, optimizer, criterion, valid_loader=None, use_GPU=use_GPU)

    predictions[target] = predict(net, test_loader, use_GPU=use_GPU)

predictions.to_csv('/home/hugoperrin/Bureau/Datasets/ToxicComment/6th_submission.csv')

time2 = time.time()
diff_time = (time2 - time1)/60
print("\n\nTraining time is {} minutes\n".format(round(diff_time,1)))