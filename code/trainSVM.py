# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
import numpy as np

from settings import lstm_settings, ROOT_DATA_PATH, TRAIN_FILE
from bert_data import df_to_dataset
from bert_trainer import BertTrainer
from utils import get_logger, set_seed
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from text_utils import normalize
from sklearn.model_selection import train_test_split
from datetime import datetime
from trainer import Trainer
from utils import device, to_indexes, pad
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torchtext import data
from tqdm import tqdm

if __name__ == '__main__':

    logger = get_logger()

    set_seed(3)

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    X_train = train_df['sentence'].values
    y_train = train_df['label'].values

    X_split = [normalize(t.split()) for t in X_train]

    X_train, X_test, y_train, y_test = train_test_split(X_split, y_train, test_size=0.2, stratify=y_train)

    text_field = data.Field()
    # build the vocabulary using train dataset
    text_field.build_vocab(X_train, max_size=10000)

    # pad
    X_train_pad = [pad(s, 128) for s in tqdm(X_train, desc='pad')]
    X_test_pad = [pad(s, 128) for s in tqdm(X_test, desc='pad')]

    # to index
    X_train_index = [to_indexes(text_field.vocab, s) for s in tqdm(X_train_pad, desc='to index')]
    X_test_index = [to_indexes(text_field.vocab, s) for s in tqdm(X_test_pad, desc='to index')]

    embed = nn.Embedding(20000,16)

    X_embed = embed(torch.tensor(X_train_index,dtype=torch.long)).squeeze(1)
    X_embed = X_embed.view(X_embed.shape[0],-1)
    print('shape of X_embed = {}'.format(X_embed.shape))
    X_embed = X_embed.cpu().detach().numpy()

    # initialise the SVM classifier
    classifier = LinearSVC()
    #classifier = SVC()
    # train the classifier
    t1 = datetime.now()
    classifier.fit(X_embed, y_train)
    print('training time = {}'.format(datetime.now() - t1))

    X_test_embed = embed(torch.tensor(X_test_index,dtype=torch.long)).squeeze(1)
    X_test_embed = X_test_embed.view(X_test_embed.shape[0],-1)
    print('shape of X_test_embed = {}'.format(X_test_embed.shape))
    X_test_embed = X_test_embed.cpu().detach().numpy()

    preds = classifier.predict(X_test_embed)
    print('SVM accuracy = {}'.format(accuracy_score(y_test, preds)))
