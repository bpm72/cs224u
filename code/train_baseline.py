# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
import sys
import os
import sst

from settings import lstm_settings, ROOT_DATA_PATH, TRAIN_FILE
from lstm_trainer import LSTMBaseline
from utils import get_logger, set_seed



if __name__ == '__main__':

    if len(sys.argv) == 0 :
        print('Usage python train_baseline <model_name>')
        print('model_names = LSTM, CNN, BERT, DISTIL-LSTM, DISTIL-CNN')

    SST_HOME = os.path.join('', 'trees')

    model_name = sys.argv[1]
    print('Model selected = {}'.format(model_name))

    logger = get_logger()

    set_seed(3)

    #train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    #X_train = train_df['sentence'].values
    #y_train = train_df['label'].values

    X_train_lst, y_train_txt = sst.build_rnn_dataset( SST_HOME, sst.train_reader, class_func=sst.binary_class_func)
    X_train = [' '.join(X_train_lst[index]) for index in range(len(X_train_lst))]

    y_train = []
    for label in y_train_txt:
        if label =='positive':
            y_train.append(1)
        else:
            y_train.append(0)

    X_test_lst, y_test_txt = sst.build_rnn_dataset( SST_HOME, sst.test_reader, class_func=sst.binary_class_func)
    X_test = [' '.join(X_test_lst[index]) for index in range(len(X_test_lst))]

    y_test = []
    for label in y_test_txt:
        if label =='positive':
            y_test.append(1)
        else:
            y_test.append(0)

    my_trainer = LSTMBaseline(lstm_settings, logger)
    model, vocab = my_trainer.train(X_train, X_test, y_train, y_test, y_train, model_name, ROOT_DATA_PATH)
