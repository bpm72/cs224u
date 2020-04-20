# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd

from settings import lstm_settings, ROOT_DATA_PATH, TRAIN_FILE
from lstm_trainer import LSTMBaseline
from utils import get_logger, set_seed

if __name__ == '__main__':

    logger = get_logger()

    set_seed(3)

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    X_train = train_df['sentence'].values
    y_train = train_df['label'].values

    my_trainer = LSTMBaseline(lstm_settings, logger)
    model, vocab = my_trainer.train(X_train, y_train, y_train, ROOT_DATA_PATH)
