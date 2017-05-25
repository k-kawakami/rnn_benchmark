# -*- coding: utf-8 -*-
"""GPU Speedtest with RNNLM."""

import argparse
import sys

import numpy
from subprocess import call

import torch
from torch import optim

from models import RNNLM
from utils import Corpus

assert torch.cuda.is_available(), 'CUDA is not available!'


parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=1234)
parser.add_argument('-cudnn', type=bool, default=1)
parser.add_argument('-tr', type=str,
                    default='data/wikitext-2/wiki.train.tokens')
parser.add_argument('-va', type=str,
                    default='data/wikitext-2/wiki.valid.tokens')
parser.add_argument('-te', type=str,
                    default='data/wikitext-2/wiki.test.tokens')
parser.add_argument('-rnn', type=str, default='LSTM',
                    help='RNN Cell Type: ')
parser.add_argument('-nlayers', type=int, default=1)
parser.add_argument('-emb_dim', type=int, default=1024)
parser.add_argument('-hid_dim', type=int, default=1024)
parser.add_argument('-tied', type=bool, default=1)
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-optimizer', type=str, default='Adam',
                    help='Optimizer Type: SGD, Adagrad, Adadelta, Adam')
parser.add_argument('-lr', type=float, default=0.0002)
parser.add_argument('-clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-seq_len', type=int, default=128)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enabled = args.cudnn

print('Python VERSION: {}'.format(sys.version))
print('pyTorch VERSION: {}'.format(torch.__version__))
print('CUDA VERSION: {}'.format(call(["nvcc", "--version"])))
print('Number CUDA Devices: {}'.format(torch.cuda.device_count()))
print('Active CUDA Device: GPU: {}'.format(torch.cuda.current_device()))
if args.cudnn:
    print('CUDNN VERSION: {}'.format(torch.backends.cudnn.version()))
else:
    print('CUDNN Disabled')

# Load data
corpus = Corpus()
corpus.build_vocab(args.tr)
Xtr, Ytr = corpus.load_data(args.tr, args.batch_size, args.seq_len)
Xva, Yva = corpus.load_data(args.va, args.batch_size, args.seq_len)
Xte, Yte = corpus.load_data(args.te, args.batch_size, args.seq_len)

# Model
model = RNNLM(args.rnn,
              args.nlayers,
              len(corpus.vocab) + 1,
              args.emb_dim,
              args.hid_dim,
              args.tied,
              args.dropout)
model.cuda()
optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

# Run!
tr_wps_record = []
va_wps_record = []
for epoch in range(args.epochs):
    tr_score, tr_wps = model.run('tr', Xtr, Ytr, args.batch_size,
                                 optimizer, args.clip)
    va_score, va_wps = model.run('va', Xva, Yva, args.batch_size)
    tr_wps_record.append(tr_wps)
    va_wps_record.append(va_wps)

    model.epoch += 1
    if va_score < model.best_score[2]:
        model.best_score = (epoch, tr_score, va_score)
        torch.save(model, 'rnnlm')

print 'Train {:.3f} wps / Eval: {:.3f}'.format(numpy.mean(tr_wps_record),
                                               numpy.mean(va_wps_record))

# Evaluate
model = torch.load('rnnlm')
model.cuda()
te_score, te_wps = model.run('te', Xte, Yte, args.batch_size)
