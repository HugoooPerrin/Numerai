
# coding: utf-8

# Import modules

import pandas as pd
import numpy as np
import time

import torch.utils.data as utils
import torch.optim as optim

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Personal modules
from models import *
from utils import *

use_GPU = torch.cuda.is_available()

time1 = time.time()

# Import data
train_comments = pd.read_csv('../../../Datasets/Numerai/w95/numerai_training_data.csv')
test_comments = pd.read_csv('../../../Datasets/Numerai/w95/numerai_tournament_data.csv')

# Preprocess data for torch
labels_train = np.array(train_comments['target'].values)
labels_train = labels_train.reshape(labels_train.shape[0], 1)

final_id = test_comments['id']

train_comments.drop(['id', 'era', 'data_type', 'target'], inplace=True, axis=1)
test_comments.drop(['id', 'era', 'data_type', 'target'], inplace=True, axis=1)

train_comments = np.array(train_comments)
test_comments = np.array(test_comments)

# Step for 1D convolution
train_comments = train_comments.reshape(train_comments.shape[0],1,train_comments.shape[1])
test_comments = test_comments.reshape(test_comments.shape[0],1,test_comments.shape[1])

time1 = time.time()

# Get final predictions
batch_size = 32
num_epoch = 6

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

# Model
net = Inception()

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimization algorithm
optimizer = optim.RMSprop(net.parameters(), lr=0.000012, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9)

# Training model
train(num_epoch, net, train_loader, optimizer, criterion, display_step=2000, valid_loader=None,
      use_GPU=use_GPU)

# Predicting
predictions = predict(net, test_loader, use_GPU=use_GPU)

# Getting the right format
predictions = pd.DataFrame(predictions, columns = ['probability'])
predictions['id'] = final_id

predictions.to_csv('../../../Datasets/Numerai/w95/1st_submission.csv')

time2 = time.time()
diff_time = (time2 - time1)/60
print("\n\nProcessing time is {} minutes\n".format(round(diff_time,1)))