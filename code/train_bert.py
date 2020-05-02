# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
from transformers import BertTokenizer

from settings import bert_settings, ROOT_DATA_PATH, TRAIN_FILE, DEV_FILE
from bert_data import df_to_dataset
from bert_trainer import BertTrainer
from utils import get_logger, set_seed
import os
import sys
import sst
import utils

if __name__ == '__main__':

    logger = get_logger()

    set_seed(3)
    
    SST_HOME = os.path.join('', 'trees')

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(DEV_FILE, encoding='utf-8', sep='\t')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    
    X_train_lst, y_train_txt = sst.build_rnn_dataset( SST_HOME, sst.train_reader, class_func=sst.binary_class_func)
    X_train = [' '.join(X_train_lst[index]) for index in range(len(X_train_lst))]

    y_train = []
    for label in y_train_txt:
        if label =='positive':
            y_train.append(1)
        else:
            y_train.append(0)

    train_df = pd.DataFrame({'sentence':X_train, 'label':y_train})

    X_test_lst, y_test_txt = sst.build_rnn_dataset( SST_HOME, sst.test_reader, class_func=sst.binary_class_func)
    X_test = [' '.join(X_test_lst[index]) for index in range(len(X_test_lst))]

    y_test = []
    for label in y_test_txt:
        if label =='positive':
            y_test.append(1)
        else:
            y_test.append(0)

    test_df = pd.DataFrame({'sentence':X_test, 'label':y_test})
    

    train_dataset = df_to_dataset(train_df, bert_tokenizer, bert_settings['max_seq_length'])
    test_dataset  = df_to_dataset(test_df, bert_tokenizer, bert_settings['max_seq_length'])

    trainer = BertTrainer(bert_settings, logger)
    model = trainer.train(train_dataset, test_dataset, bert_tokenizer, ROOT_DATA_PATH)

