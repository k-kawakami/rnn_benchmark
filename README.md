# rnn_benchmark

This is a GPU speed test with standard language model.
All the experiments are done with Pytorch & cudnn backend.

# Evaluation

The evaluations are done on a standard language modeling dataset, Wikitext-2.

# Model Configuration

Since we are not trying to obtain the SoTA result, we used large minibatch size and gradient clipping to consume large amount of GPU memory.

```
-rnn LSTM 
-nlayers 1 
-emb_dim 1024 
-hid_dim 1024 
-tied 0 
-epochs 100 
-optimizer Adam 
-lr 0.002 
-clip 0.25 
-dropout 0.5 
-batch_size 128 
-seq_len 128
```
