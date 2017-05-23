import os
import sys

import codecs
import timeit

import numpy

import torch
from torch.autograd import Variable


class Corpus:
    def __init__(self):
        self.vocab = set()

    def build_vocab(self, file_name):
        data = self._load_data(file_name)
        self.vocab = set(data)
        self.i2x = {i:x for i, x in enumerate(self.vocab)}
        self.x2i = {x:i for i, x in self.i2x.items()}

    def load_data(self, file_name, batch_size, seq_len):
        data = self._load_data(file_name)
        x = numpy.array([ self.x2i[x] for x in data])
        X, Y = self.reorder(x, batch_size, seq_len)
        return torch.from_numpy(X).cuda(),\
               torch.from_numpy(Y).cuda()


    def _load_data(self, file_name):
        """
        Loads Penn Tree files downloaded from https://github.com/wojzaremba/lstm
        Parameters
        ----------
        file_name : str
            Path to Penn tree file.
        vocab_map : dictionary
            Dictionary mapping words to integers
        vocab_idx : one element list
            Current vocabulary index.
        Returns
        -------
        Returns an array with each words specified in file_name as integers.
        Note that the function has the side effects that vocab_map and vocab_idx
        are updated.
        Notes
        -----
        This is python port of the LUA function load_data in
        https://github.com/wojzaremba/lstm/blob/master/data.lua
        borrowed from
        https://github.com/GabrielPereyra/norm-rnn/blob/master/datasets.py
        """
        def process_line(line):
            line = line.lstrip()
            line = line.replace('\n', '<eos>')
            words = line.split(" ")
            if words[-1] == "":
                del words[-1]
            return words

        words = []
        with codecs.open(file_name, 'r', 'utf_8_sig') as f:
            for line in f.readlines():
                words += process_line(line)

        n_words = len(words)
        print("Loaded %i words from %s" % (n_words, file_name))
        return words


    def reorder(self, x_in, batch_size, model_seq_len):
        """
        Rearranges data set so batches process sequential data.
        If we have the dataset:
        x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        and the batch size is 2 and the model_seq_len is 3. Then the dataset is
        reordered such that:
                       Batch 1    Batch 2
                     ------------------------
        batch pos 1  [1, 2, 3]   [4, 5, 6]
        batch pos 2  [7, 8, 9]   [10, 11, 12]
        This ensures that we use the last hidden state of batch 1 to initialize
        batch 2.
        Also creates targets. In language modelling the target is to predict the
        next word in the sequence.
        Parameters
        ----------
        x_in : 1D numpy.array
        batch_size : int
        model_seq_len : int
            number of steps the model is unrolled
        Returns
        -------
        reordered x_in and reordered targets. Targets are shifted version of x_in.
        borrowed from
        https://github.com/GabrielPereyra/norm-rnn/blob/master/datasets.py
        """
        if x_in.ndim != 1:
            raise ValueError("Data must be 1D, was", x_in.ndim)

        if x_in.shape[0] % (batch_size*model_seq_len) == 0:
            print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
                  "set to x_in = x_in[:-1]")
            x_in = x_in[:-1]

        x_resize =  \
            (x_in.shape[0] // (batch_size*model_seq_len))*model_seq_len*batch_size
        n_samples = x_resize // (model_seq_len)
        n_batches = n_samples // batch_size

        targets = x_in[1:x_resize+1].reshape(n_samples, model_seq_len)
        x_out = x_in[:x_resize].reshape(n_samples, model_seq_len)

        out = numpy.zeros(n_samples, dtype=int)
        for i in range(n_batches):
            val = range(i, n_batches*batch_size+i, n_batches)
            out[i*batch_size:(i+1)*batch_size] = val

        x_out = x_out[out]
        targets = targets[out]
        return x_out.astype('int64'), targets.astype('int64')


class ProgressBar(object):
    def __init__(self, mode, epoch, nbatches):
        self.mode = mode
        self.epoch = epoch

        self.size = 60
        self.nbatches = nbatches

        self.done = 0
        self.left = self.size
        self.batch = 0

        self.begin = timeit.default_timer()

    def __str__(self):
        report = '{} {}: EPOCH {} [{}{}] '.format(os.getpid(), self.mode, str(self.epoch).zfill(3), '#' * self.done, '.' * self.left)
        for name, result, f in self.report_list:
            report += '{}({:.3f}) '.format(name, f(result))
        report += 'time({:.2f}s)\r'.format(timeit.default_timer() - self.begin)
        return report

    def update(self, report_list):
        #[(name, result, func), ... (pp, [...., lambda x: numpy.exp(numpy.mean(x)))]
        self.report_list = report_list

        self.batch += 1
        self.done = int(self.size * self.batch) / self.nbatches
        self.left = self.size - self.done

        sys.stdout.write(str(self))
        sys.stdout.flush()
