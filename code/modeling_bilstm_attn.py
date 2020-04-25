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
                           batch_first=True,
                           dropout=dropout)

        self.embed_drop = nn.Dropout(0.5)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

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

        ##print(' size of input text = {}'.format(text.shape))
        x = self.embedding(text.t())
        #x = self.embed_drop(x)
        ##print('size of embed out = {}'.format(x.shape))

        #r, _ = self.rnn(x)  ## (32,128,16)
        #r, _ = self.rnn(x.transpose(0,1))
        lstm_output, (last_hidden_state, last_cell_state) = self.rnn(x)
        #final_hidden, cell = last_hidden
        ##print('size of last_hidden_state = {}'.format(last_hidden_state.shape))
        ##print('size of lstm_output = {}'.format(lstm_output.shape))
        #r = lstm_output
        #r1 = (r[:,:,:self.hidden_dim] + r[:,:,self.hidden_dim:])
        #r2 = (r[:,:,:self.hidden_dim] - r[:,:,self.hidden_dim:])
        #r = torch.cat([r1,r2],dim=0)
        #print('size of r = {}'.format(r.shape))

        # attention
        y = torch.tanh(lstm_output)
        ##print('size of y = {}'.format(y.shape))

        att = self.fc_att(y)  # [b,msl,h*2]->[b,msl]
        ##print('size of att = {}'.format(att.shape))
        alpha = F.softmax(att,dim=1).transpose(1,2) # [b,msl]
        ##print('size of alpha = {}'.format(alpha.shape))
        #r_att = torch.sum(att.unsqueeze(-1) * r, dim=1)  # [b,h*2]
        r_att = alpha.bmm(lstm_output).squeeze(1)
        h = torch.tanh(r_att)
        ##print('size of att final = {}'.format(h.shape)) # (batch_size, hidden_size)

        # pooling
        # r_avg = mask_mean(r, mask)  # [b,h*2]
        # r_max = mask_max(r, mask)  # [b,h*2]
        # r = torch.cat([r_avg, r_max, r_att], dim=-1)  # [b,h*6]

        # feed-forward & dropout
        #logits = self.fc(r_att)  # [b,h*2]->[b,h]
        ##print('size of logits b4 drop = {}'.format(logits.shape))
        #logits = self.dropout(logits)  # [b,h*2]->[b,h]
        ##print('size of logits = {}'.format(logits.shape))

        #hidden = self.dropout(torch.cat((final_hidden[-2, :, :], final_hidden[-1, :, :]), dim=1))
        #hidden = self.dropout(torch.cat((last_hidden_state[-2], last_hidden_state[-1]), dim=1))
        logits = self.fc(h)
        logits = self.dropout(logits)
        ##print('size of logits = {}'.format(logits.shape))

        return logits


        #hidden, cell = hidden
        #hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        #x = self.fc(hidden)
        #return x
