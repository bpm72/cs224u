# coding: utf-8

from __future__ import unicode_literals, print_function

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from settings import distillation_settings, TRAIN_FILE, ROOT_DATA_PATH, DEV_FILE
from settings import bert_settings
from bert_data import df_to_dataset
from bert_trainer import batch_to_inputs
from lstm_trainer import LSTMDistilled
from utils import get_logger, device, set_seed

import os
import sst
import utils


if __name__ == '__main__':
    logger = get_logger()

    if len(sys.argv) == 0 :
        print('Usage python distil_bert.py <model_name>')
        print('model_names = LSTM, CNN, BERT, DISTIL-LSTM, DISTIL-CNN')

    SST_HOME = os.path.join('', 'trees')

    model_name = sys.argv[1]
    print('Model selected = Distil_{}'.format(model_name))

    set_seed(3)

    '''
    # 1. get data
    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_model = BertForSequenceClassification.from_pretrained(ROOT_DATA_PATH)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = df_to_dataset(train_df, tokenizer, distillation_settings['max_seq_length'])
    sampler = SequentialSampler(train_dataset)
    data = DataLoader(train_dataset, sampler=sampler, batch_size=distillation_settings['train_batch_size'])

    bert_model.to(device())
    bert_model.eval()

    bert_logits = None

    for batch in tqdm(data, desc="bert logits"):
        batch = tuple(t.to(device()) for t in batch)
        inputs = batch_to_inputs(batch)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            _, logits = outputs[:2]

            #print('shape of bert output = {}'.format(logits.shape))
            logits = logits.cpu().numpy()
            if bert_logits is None:
                bert_logits = logits
            else:
                bert_logits = np.vstack((bert_logits, logits))

    # Its important to use binary mode 
    dbfile = open('bert_logits_pickle_new.pkl', 'wb') 
      
    # source, destination 
    pickle.dump(bert_logits, dbfile)                      
    dbfile.close() 

    print("Completed dumping bert logits in 'bert_logits_pickle.pkl' ")
    '''

    # Read the stored bert_logits. 
    dbfile = open('bert_logits_pickle_new.pkl', 'rb')      
    bert_logits = pickle.load(dbfile) 
    dbfile.close() 

    # 2.
    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    X_train = train_df['sentence'].values
    y_train = train_df['label'].values

    test_df = pd.read_csv(DEV_FILE, encoding='utf-8', sep='\t')
    X_test = test_df['sentence'].values
    y_test = test_df['label'].values

    # 3. trainer
    distiller = LSTMDistilled(distillation_settings, logger)

    # 4. train
    model, vocab = distiller.train(X_train, X_test, y_train, y_test, bert_logits, sys.argv[1], ROOT_DATA_PATH)

