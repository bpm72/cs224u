# Final Project
  CS224u, Stanford, Spring 2020 <br />
  Instructor - Prof Christopher Potts

## Instructions to run

Run the baseline BERT model fine-tuned on the SST-2 database<br />
$ python train_bert.py<br />

Run the baseline student models to set the baseline performance<br />
$ python train_baseline.py LSTM <br />
$ python train_baseline.py BiLSTM_Attn<br /> 
$ python train_baseline.py CNN <br />

Now, run the models with distillation to evaluate performance<br />
$ python distil_bert.py LSTM <br />
$ python distil_bert.py BiLSTM_Attn<br /> 
$ python distil_bert.py CNN <br />
