# Final Project
  CS224u, Stanford, Spring 2020 <br />
  Instructor - Prof Christopher Potts

## Instructions to run

Run the baseline BERT model fine-tuned on the SST-2 database
$ python bert_trainer.py

Run the baseline student models to set the baseline performance
$ python train_baseline.py LSTM 
$ python train_baseline.py BiLSTM_Attn 
$ python train_baseline.py CNN 

Now, run the models with distillation to evaluate performance
$ python distil_bert.py LSTM 
$ python distil_bert.py BiLSTM_Attn 
$ python distil_bert.py CNN 
