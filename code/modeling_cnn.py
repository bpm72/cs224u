# coding: utf-8
from __future__ import unicode_literals, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.svm import LinearSVC

class SimpleCNN(nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_classes, num_kernels,
                 kernel_sizes_lst, dropout,batch_size, device=None):
        super(SimpleCNN, self).__init__()

        self.batch_size = batch_size
        self.V = input_dim
        self.D = embedding_dim
        self.C = num_classes
        self.Ci = 1
        self.Co = len(kernel_sizes_lst)
        self.Ks = kernel_sizes_lst

        self.embed = nn.Embedding(self.V, self.D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (K, self.D)) for K in self.Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.fc1 = nn.Linear(len(self.Ks)*self.Co, self.C)

        self.dropout = nn.Dropout(dropout)
        #self.sigmoid = nn.Sigmoid()
        self.device = self.init_device(device)


    @staticmethod
    def init_device(device):
        if device is None:
            return torch.device('cuda')
        return device

    def forward(self, x, text_lengths=None):
        x = self.embed(x.t())  # (N, W, D)
        
#        if self.args.static:
#        x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        #output = self.sigmoid(logit)

        return logit


class SimpleSVM(object):

    def __init__(self, input_dim, embedding_dim,output_dim):
        super(self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.SVMclassifier = LinearSVC() 

    def forward(self, text, text_lengths=None):
        embedded = self.embedding(text)
        classifier.fit(embedded.flatten(), y_train2)
        preds = self.SVMclassifier.predict(X_test2)

        return self.fc(hidden.squeeze(0))

    @staticmethod
    def probs_to_logits(probs, is_binary=False):
        r"""
        Converts a tensor of probabilities into logits. For the binary case,
        this denotes the probability of occurrence of the event indexed by `1`.
        For the multi-dimensional case, the values along the last dimension
        denote the probabilities of occurrence of each of the events.
        """
        ps_clamped = clamp_probs(probs)
        if is_binary:
            return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
        return torch.log(ps_clamped)