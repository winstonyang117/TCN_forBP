import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
#import pywt
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torchinfo import summary
from torchTrainingFuc import train_loop, test_loop
from DataLoader import GetLoader
from scipy import signal
import timeit
import torch.nn.functional as F
from sklearn.model_selection import train_test_split 

batch_size = 64
epochs = 500
lr = 0.001 # initialize learning rate 

def data_processing(data_train, data_test, batch_size):
    # Data preprocessing: Filtering, smoothing, training&tesing data, etc.  
    train_dataset = data_train
    test_dataset = data_test

    #resplit mix data
    X_train = train_dataset[:, :-3]
    Y_train = train_dataset[:, -2]
    X_test = test_dataset[:, :-3]
    Y_test = test_dataset[:, -2]

    X_training = np.expand_dims(X_train, axis = -2)
    X_testing = np.expand_dims(X_test, axis = -2)
    
    training_data = GetLoader(X_training.astype(np.float32), Y_train.astype(np.float32))
    testing_data = GetLoader(X_testing.astype(np.float32), Y_test.astype(np.float32))

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, X_training

def data_processing_2(raw_data, batch_size):
    # Data preprocessing: Filtering, smoothing, training&tesing data, etc.  
    labels = raw_data[:,-2]
    X_train, X_test, Y_train, Y_test = train_test_split (raw_data,labels, test_size=0.3, train_size=0.7 )

    X_training = np.expand_dims(X_train, axis = -2)
    X_testing = np.expand_dims(X_test, axis = -2)
    
    training_data = GetLoader(X_training.astype(np.float32), Y_train.astype(np.float32))
    testing_data = GetLoader(X_testing.astype(np.float32), Y_test.astype(np.float32))

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader, X_training

def lr_scheduler(epoch, lr):
    if epoch % 200 == 0 and epoch != 0:
        lr = lr * 0.5
        print("lr changed to {}".format(lr))
    return lr
 
test_mae_list = []


# Modify the data size after convlution performed, make it the same as the previous input data.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class hr_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1):
        super(hr_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=2, padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=1, padding = int(dilation * (kernel_size -1) / 2), 
                                            dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, stride=2) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(n_outputs)
        self.dropout3 = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, pad):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out=F.pad(input=out,pad=pad)
        # print(out.shape)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        # print(out.shape)
        res = x if self.downsample is None else self.downsample(x)
        res = out + res
        res = self.batchnorm(res)
        res = self.relu(res)
        res = self.dropout3(res)
        return res

#HR model
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, first_output_num_channels, num_blocks, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            dilation_size = 2 ** i  
            in_channels = num_inputs if i == 0 else first_output_num_channels * (2 ** (i-1)) 
            out_channels = first_output_num_channels if i == 0 else int(in_channels * 2)
            self.layers += [hr_TemporalBlock(in_channels, out_channels, kernel_size, 
                                          dilation=dilation_size, dropout=dropout)]
        self.flatten = nn.Flatten()
        self.regressor = nn.Linear(8192, 1) ################# later
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for i in range(len(self.layers)):
            n_padding = 2 ** i
            pad = (0, n_padding)
            x = self.layers[i](x, pad)
            # print('x shape is '+str(x.shape))
           
        x = self.flatten(x)
        x = self.regressor(x)
        return x

# Loading data
raw_data = np.load('CTRU_good_data_2022.npy')
# data_train = np.load('sscg_longPeriod_train.npy') #
# data_test = np.load('sscg_longPeriod_test.npy') #
# print('Loaded data size: '+str(data_train.shape))
print('Loaded data size: '+str(raw_data.shape))
# train_dataloader, test_dataloader, X_training = data_processing(data_train, data_test, batch_size)
train_dataloader, test_dataloader, X_training = data_processing_2(raw_data, batch_size)

# start training
print('===============================')
print('start training......')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TemporalConvNet(num_inputs = 1, first_output_num_channels = 16, num_blocks = 5, kernel_size=3, dropout=0.2)
print('x_training size: '+str(X_training.shape))

model.to(device)
print(device)
summary(model, (batch_size, X_training.shape[-2], X_training.shape[-1]), verbose = 1)


is_loss_MAE = 1
loss_fn = torch.nn.MSELoss()
metric_fn = torch.nn.L1Loss()

for t in range(epochs):
    if t == 0:
        lrate = lr_scheduler(t, lr)
        print("This is the "+ str(t) + "th epoch:")
        train_loop(train_dataloader, model, device, lrate, metric_fn, loss_fn)
        test_mae = test_loop(test_dataloader, model, t, loss_fn, metric_fn, device)
        test_mae_list.append(test_mae)
    else: 
        lrate = lr_scheduler(t, lrate)
        print("This is the "+ str(t) + "th epoch:")
        train_loop(train_dataloader, model, device, lrate, metric_fn, loss_fn)
        test_mae = test_loop(test_dataloader, model, t, loss_fn, metric_fn, device)
        test_mae_list.append(test_mae)

print('Test mae_list size is: '+str(len(test_mae_list)))
print(str(test_mae_list.index(min(test_mae_list)))+'th epoch has the minimum test MAE: '+str(min(test_mae_list)))

test_mae_sscg = np.array(test_mae_list)
np.save('test_mae_sscg.npy',test_mae_sscg)   

print("Training Done!")

