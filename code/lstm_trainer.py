# coding: utf-8

from __future__ import unicode_literals, print_function

import os

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torchtext import data
from tqdm import tqdm
import pandas as pd

from loss import WeightedMSE
from modeling_lstm import SimpleLSTM
from modeling_bilstm_attn import BiLSTM_Attn
from modeling_cnn import SimpleCNN
from naiveBert import BasicTransformer
from text_utils import normalize
from trainer import Trainer
from utils import device, to_indexes, pad
from sklearn.metrics import classification_report


class _LSTMBase(Trainer):

    vocab_name = None
    weights_name = None

    def model(self, text_field):
        raise NotImplementedError()

    @staticmethod
    def to_dataset(x, y, y_real):
        torch_x = torch.tensor(x, dtype=torch.long)
        torch_y = torch.tensor(y, dtype=torch.float)
        torch_real_y = torch.tensor(y_real, dtype=torch.long)
        print('size of torch_x = {}'.format(torch_x.shape))
        print('size of torch_y = {}'.format(torch_y.shape))
        print('size of torch_real_y = {}'.format(torch_real_y.shape))
        return TensorDataset(torch_x, torch_y, torch_real_y)

    @staticmethod
    def to_device(text, bert_prob, real_label):
        text = text.to(device())
        bert_prob = bert_prob.to(device())
        real_label = real_label.to(device())
        return text, bert_prob, real_label

    def train(self, X_train, X_test, y_train, y_test, bert_prob, model_name, output_dir):

        X_train = [normalize(t.split()) for t in X_train]
        X_test  = [normalize(t.split()) for t in X_test]

        #split_arrays = train_test_split(X_split, y, y_real, test_size=self.settings['test_size'], stratify=y_real)
        #X_train, X_test, y_train, y_test, y_real_train, y_real_test = split_arrays
        #X_train, X_test, y_train, y_test, y_real_train, y_real_test = split_arrays
        y_real_train = y_train
        y_real_test = y_test

        text_field = data.Field()

        # build the vocabulary using train dataset
        text_field.build_vocab(X_train, max_size=30000)

        # pad
        X_train_pad = [pad(s, self.settings['max_seq_length']) for s in tqdm(X_train, desc='pad')]
        X_test_pad = [pad(s, self.settings['max_seq_length']) for s in tqdm(X_test, desc='pad')]

        # to index
        X_train_index = [to_indexes(text_field.vocab, s) for s in tqdm(X_train_pad, desc='to index')]
        X_test_index = [to_indexes(text_field.vocab, s) for s in tqdm(X_test_pad, desc='to index')]

        train_dataset = self.to_dataset(X_train_index, bert_prob, y_real_train)
        val_dataset = self.to_dataset(X_test_index, y_test, y_real_test)

        model = self.model(text_field,model_name)
        model.to(device())

        self.full_train(model, model_name, train_dataset, val_dataset, output_dir)
        torch.save(text_field, os.path.join(output_dir, self.vocab_name))

        return model, text_field.vocab

    @staticmethod
    def optimizer(model):
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return optimizer, scheduler

    def full_train(self, model, model_name, train_dataset, val_dataset, output_dir):
        """
        :param model:
        :param train_dataset:
        :param val_dataset:
        :param output_dir:
        :return:
        """
        train_settings = self.settings
        num_train_epochs = train_settings['num_train_epochs']

        best_eval_loss = 100000

        for epoch in range(num_train_epochs):
            train_loss = self.epoch_train_func(model, model_name, train_dataset)
            print('starting evaluate() function')
            eval_loss, acc = self.epoch_evaluate_func(model, val_dataset, epoch)
            self.log_epoch(train_loss, eval_loss, acc, epoch)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.logger.info('save best model {:.4f}'.format(eval_loss))
                torch.save(model.state_dict(), os.path.join(output_dir, self.weights_name))

    def epoch_train_func(self, model, model_name, dataset):
        train_loss = 0
        train_sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=self.settings['train_batch_size'], drop_last=True)
        model.train() #put the model in train mode - dont confuse with train() method defined above

        p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info('# of trainable params {}'.format(p_count))
        print('# of trainable params {}'.format(p_count))

        num_examples = 0
        optimizer, scheduler = self.optimizer(model)
        for i, (text, ldr_bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Train')):
            text, ldr_bert_prob, real_label = self.to_device(text, ldr_bert_prob, real_label)
            model.zero_grad()
            #print('model input shape : {}'.format(text.t().shape))
            output = model(text.t(), model_name).squeeze(1)
            #print('output shape : {}'.format(output.shape))
            #print('ldr_bert_prob shape : {}'.format(ldr_bert_prob.shape))
            #print('real_label shape : {}'.format(real_label.shape))
            loss = self.loss(output, ldr_bert_prob, real_label)
            #print('exiting loss()')

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_examples += len(real_label)
        scheduler.step()
        return train_loss / num_examples

    def epoch_evaluate_func(self, model, eval_dataset, epoch):
        eval_sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, 
                                 sampler=eval_sampler,
                                 batch_size=self.settings['eval_batch_size'],
                                 drop_last=True)

        eval_loss = 0.0
        acc = 0.0
        num_examples = 0
        model.eval()
        predictions = None
        labels = None
        for i, (text, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Val')):
            text, bert_prob, real_label = self.to_device(text, bert_prob, real_label)
            output = model(text.t()).squeeze(1)

            loss = self.eval_loss(output, bert_prob, real_label)
            eval_loss += loss.item()

            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1)
            acc += torch.sum(pred_label == real_label).cpu().numpy()
            num_examples += len(real_label)

            predictions, labels = self.stack(predictions, labels, probs, real_label)

        self.log_pr(labels, predictions, epoch)
        #print('plabels = {}'.format(labels))
        #print('predictions = {} '.format(predictions))
        #print(classification_report(real_label.cpu().numpy(), pred_label.cpu().numpy()))
        return eval_loss / num_examples, acc / num_examples

    def loss(self, output, bert_prob, real_label):
        raise NotImplementedError()


class LSTMBaseline(_LSTMBase):
    """
    LSTM baseline
    """

    vocab_name = 'text_vocab.pt'
    weights_name = 'simple_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMBaseline, self).__init__(settings, logger)

        self.criterion = torch.nn.CrossEntropyLoss()

    def loss(self, output, bert_prob, real_label):
        #print('output.shape = {}'.format(output.shape))
        #print('real_label.shape = {}'.format(real_label.shape))
        return self.criterion(output, real_label)

    def eval_loss(self, output, bert_prob, real_label):
        return self.criterion(output, real_label)

    def model(self, text_field, model_name):
        if model_name == 'LSTM':
            print('LSTM Model chosen')
            model = SimpleLSTM(
                input_dim=len(text_field.vocab),
                embedding_dim=32,
                hidden_dim=32,
                output_dim=2,
                n_layers=1,
                bidirectional=True,
                dropout=0.5,
                batch_size=self.settings['train_batch_size'])
            return model
        elif model_name == 'BiLSTM_Attn':
            print('BiLSTM Attention Model chosen')
            model = BiLSTM_Attn(
                input_dim=len(text_field.vocab),
                embedding_dim=32,
                hidden_dim=32,
                output_dim=2,
                n_layers=1,
                bidirectional=True,
                dropout=0.5,
                batch_size=self.settings['train_batch_size'])
            return model
        elif model_name == 'CNN':
            print('CNN Model chosen')
            model = SimpleCNN(
                input_dim=len(text_field.vocab),
                embedding_dim=32,
                hidden_dim=32,
                num_classes=2,
                num_kernels=3,
                kernel_sizes_lst=[3,4,5],
                dropout=0.5,
                batch_size=self.settings['train_batch_size'])
            return model
        else :
            print('unknown model chose')
            exit(0)

#        model = BasicTransformer(
#            input_dim=len(text_field.vocab),
#            embedding_dim=16,
#            heads = 4, 
#            mask=False, 
#            seq_length=128, 
#            ff_hidden_mult=4, 
#            dropout=0.2)
#        return model

class LSTMDistilled(_LSTMBase):
    """
    LSTM distilled
    """

    vocab_name = 'distil_text_vocab.pt'
    weights_name = 'distil_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMDistilled, self).__init__(settings, logger)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss() #WeighterMSE() 
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.alpha = 0.5

    def loss(self, output, bert_prob, real_label):
        #return self.alpha * self.criterion_ce(output, real_label) + (1 - self.alpha) * self.criterion_mse(output, bert_prob)
        l1 = self.criterion_ce(output, real_label)
        #print('format of output = {}'.format(output.shape))
        #print('format of bert_prob = {}'.format(bert_prob.shape))
        l2 = self.criterion_mse(output,bert_prob)
        return self.alpha*l1 + (1-self.alpha)*l2
 
    def eval_loss(self, output, bert_prob, real_label):
        return self.criterion(output, real_label)

    def model(self, text_field, model_name):
        if model_name == 'LSTM':
            model = SimpleLSTM(
                input_dim=len(text_field.vocab),
                embedding_dim=32,
                hidden_dim=32,
                output_dim=2,
                n_layers=1,
                bidirectional=True,
                dropout=0.3,
                batch_size=self.settings['train_batch_size'])
            return model
        elif model_name == 'BiLSTM_Attn':
            print('BiLSTM Attention Model chosen')
            model = BiLSTM_Attn(
                input_dim=len(text_field.vocab),
                embedding_dim=32,
                hidden_dim=32,
                output_dim=2,
                n_layers=1,
                bidirectional=True,
                dropout=0.3,
                batch_size=self.settings['train_batch_size'])
            return model
        elif model_name == 'CNN':
            model = SimpleCNN(
                input_dim=len(text_field.vocab),
                embedding_dim=16,
                hidden_dim=8,
                num_classes=2,
                num_kernels=2,
                kernel_sizes_lst=[3,4],
                dropout=0.3,
                batch_size=self.settings['train_batch_size'])
            return model
        else :
            print('unknown model chose')
            exit(0)

class LSTMDistilledWeighted(LSTMDistilled):
    """
    LSTM distilled with weighted MSE
    """
    vocab_name = 'w_distil_text_vocab.pt'
    weights_name = 'w_distil_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMDistilledWeighted, self).__init__(settings, logger)
        self.criterion_mse = WeightedMSE()

    def loss(self, output, bert_prob, real_label):
        l1 = self.a * self.criterion_ce(output, real_label)
        l2 = (1 - self.a) * self.criterion_mse(output, bert_prob, real_label)
        return l1 + l2
