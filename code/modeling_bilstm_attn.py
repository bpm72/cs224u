# coding: utf-8

from __future__ import unicode_literals, print_function

import torch
from torch import nn
import torch.nn.functional as F

class BiLSTM_Attn(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_size, device=None):
        super(BiLSTM_Attn, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.LSTM(self.embedding.embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.embed_drop = nn.Dropout(0.3)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.drop = nn.Dropout(dropout)

        self.device = self.init_device(device)

    @staticmethod
    def init_device(device):
        if device is None:
            return torch.device('cuda')
        return device

    def forward(self, text, text_lengths=None):

        # mask
        #max_seq_len = torch.max(seq_len)
        #mask = seq_mask(seq_len, max_seq_len)  # [b,msl]

        ##print('size of text = {}'.format(text.shape))
        x = self.embedding(text.t())
        #x = self.embed_drop(x)
        #print('type(x) =  {}'.format(type(x)))
        ##print('size of x = {}'.format(x.shape))

        #r, hidden = self.rnn(x)
        #r, _ = self.rnn(x.transpose(0,1))
        r, _ = self.rnn(x)
        ##print('size of r = {}'.format(r.shape))
        #r = r[:,:,:self.hidden_dim] + r[:,:,self.hidden_dim:]
        #print('size of r = {}'.format(r.shape))

        # attention
        att = self.fc_att(r)  # [b,msl,h*2]->[b,msl]
        ##print('size of att = {}'.format(att.shape))
        att = F.softmax(att,dim=1).transpose(1,2)  # [b,msl]
        ##print('size of att after softmax = {}'.format(att.shape))
        #r_att = torch.sum(att.unsqueeze(-1) * r, dim=1)  # [b,h*2]
        r_att = att.bmm(r).squeeze(1)
        h = torch.tanh(r_att)
        ##print('size of att final = {}'.format(h.shape)) # (batch_size, hidden_size)

        # pooling
        # r_avg = mask_mean(r, mask)  # [b,h*2]
        # r_max = mask_max(r, mask)  # [b,h*2]
        # r = torch.cat([r_avg, r_max, r_att], dim=-1)  # [b,h*6]

        # feed-forward & dropout
        logits = self.fc(h)  # [b,h*2]->[b,h]
        ##print('size of logits b4 drop = {}'.format(logits.shape))
        logits = self.drop(logits)  # [b,h*2]->[b,h]
        ##print('size of logits = {}'.format(logits.shape))

        return logits


        #hidden, cell = hidden
        #hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        #x = self.fc(hidden)
        #return x
