# coding: utf-8

from __future__ import unicode_literals, print_function

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

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
                           batch_first=True,
                           dropout=dropout)

        self.embed_drop = nn.Dropout(0.5)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)
        #self.fc_att = nn.Linear(hidden_dim , 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.device = self.init_device(device)
        self.hidden = self.init_hidden()

    @staticmethod
    def init_device(device):
        if device is None:
            return torch.device('cuda')
        return device

    def init_hidden(self):
        return (Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)),
                Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)))

    def forward(self, text, text_lengths=None):

        self.hidden = self.init_hidden()

        ###print(' size of input text = {}'.format(text.shape))
        x = self.embedding(text.t())
        x = self.embed_drop(x)
        ###print('size of embed out = {}'.format(x.shape))

        lstm_output, (last_hidden_state, last_cell_state) = self.rnn(x,self.hidden)
        ###print('size of lstm_output = {}'.format(lstm_output.shape))
        #lstm_output = lstm_output[:,:,:self.hidden_dim] + lstm_output[:,:,self.hidden_dim:] # (batch_size, max_len, hidden_size)
        ###print('size of last_hidden_state = {}'.format(last_hidden_state.shape))
        ###print('size of lstm_output_merged = {}'.format(lstm_output.shape))

        # attention
        y = torch.tanh(lstm_output)
        ###print('size of y = {}'.format(y.shape))

        y = self.fc_att(y).squeeze(2)  # [b,msl,h*2]->[b,msl]
        ###print('size of y (attn) = {}'.format(y.shape))
        alpha = F.softmax(y,dim=1).unsqueeze(1) # [b,msl]
        ###print('size of alpha = {}'.format(alpha.shape))
        r_att = alpha.bmm(lstm_output).squeeze(1)
        ###print('size of r_att = {}'.format(r_att.shape)) # (batch_size, hidden_size)
        h = torch.tanh(r_att)
        ###print('size of att final = {}'.format(h.shape)) # (batch_size, hidden_size)

        logits = self.fc(h)
        logits = self.dropout(logits)
        ###print('size of logits = {}'.format(logits.shape))
        ###exit(0)

        return logits
