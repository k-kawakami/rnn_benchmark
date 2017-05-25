"""Basic RNNLM."""

import time

import numpy

import torch
from torch import nn
from torch.autograd import Variable

from utils import ProgressBar


class RNNLM(nn.Module):

    def __init__(self, rnn_type, nlayers, vocab_size,
                 emb_dim, hid_dim, tied, dropout):
        super(RNNLM, self).__init__()

        self.epoch = 0
        self.best_score = (0, float('inf'), float('inf'))

        self.proj = nn.Sequential(
                        nn.Embedding(vocab_size, emb_dim),
                        nn.Dropout(dropout)
                    )

        self.rnn = getattr(nn, rnn_type)(emb_dim,
                                         hid_dim,
                                         nlayers,
                                         bias=True,
                                         bidirectional=False,
                                         dropout=dropout,
                                         batch_first=True)
        self.logit = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(hid_dim, vocab_size)
                        )

        if tied:
            self.logit[1].weight = self.proj[0].weight

    def reset_states(self, batch_size):
        weight = next(self.parameters()).data

        if self.rnn.__class__.__name__ == 'LSTM':
            num_states = 2
        else:
            num_states = 1
        num_layers = self.rnn.num_layers

        self.states = []
        for _ in range(num_states):
            self.states.append(
                Variable(
                    weight.new(
                        num_layers,
                        batch_size,
                        self.rnn.hidden_size).zero_()
                )
            )

    def set_states(self, states):
        for i, state in enumerate(states):
            self.states[i] = Variable(state.data)

    def forward(self, x):
        x = self.proj(x)
        x, states = self.rnn(x, self.states)
        self.set_states(states)
        x = self.logit(x.contiguous().view(x.size(0) * x.size(1), x.size(2)))
        return x

    def run(self, mode, X, Y, batch_size, optimizer=None, clip=None):
        self.reset_states(batch_size)
        if optimizer:
            self.train(True)
        else:
            self.eval()

        nbatches = X.size(0) // batch_size

        pb = ProgressBar(mode, self.epoch, nbatches)
        _total_time = 0
        _total_loss = 0
        _total_word = 0

        L = nn.CrossEntropyLoss(size_average=False)

        for index in range(nbatches):
            begin = index * batch_size
            end = begin + batch_size

            # Start
            if optimizer:
                start = time.time()
                x = Variable(X[begin:end], requires_grad=False)
                t = Variable(Y[begin:end], requires_grad=False)
            else:
                start = time.time()
                x = Variable(X[begin:end], requires_grad=False, volatile=True)
                t = Variable(Y[begin:end], requires_grad=False, volatile=True)
            y = self(x)
            loss = L(y, t.view(-1))

            if optimizer:
                if clip:
                    torch.nn.utils.clip_grad_norm(self.parameters(), clip)
                self.zero_grad()
                loss.backward()
                optimizer.step()
            # End
            time_per_batch = time.time() - start
            _total_time += time_per_batch
            _total_loss += loss.cpu().data.numpy()[0]
            _total_word += float(numpy.prod(t.size()))
            pb.update([
                    ('ppl', numpy.exp(_total_loss / _total_word), lambda x: x),
                    ('wps', _total_word / _total_time, lambda x: x)
            ])

        print
        return numpy.exp(_total_loss / _total_word), _total_word / _total_time
