#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from collections import defaultdict
from sys import stdin
import os
import numpy as np

import torch
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()


class Dataset(object):
    CONSTANTS = {
        'PAD': 0,
        'SOS': 1,
        'EOS': 2,
        'UNK': 3,
    }

    def __init__(self, cfg, stream=None):
        self.cfg = cfg
        self.src_vocab = defaultdict(lambda: len(self.src_vocab))
        if self.cfg.share_vocab:
            self.tgt_vocab = self.src_vocab
        else:
            self.tgt_vocab = defaultdict(lambda: len(self.tgt_vocab))
        for k, v in Dataset.CONSTANTS.items():
            self.src_vocab[k] = v
            self.tgt_vocab[k] = v
        self.load_data_from_stream(stream=stream)
        self.update_config()

    def update_config(self):
        self.cfg.input_size = len(self.src_vocab)
        self.cfg.output_size = len(self.tgt_vocab)

    def src_lookup(self, ch, frozen=False):
        if frozen is False:
            return self.src_vocab[ch]
        return self.src_vocab.get(ch, Dataset.CONSTANTS['UNK'])

    def tgt_reverse_lookup(self, idx):
        try:
            self.tgt_inv_vocab
        except AttributeError:
            self.tgt_inv_vocab = {
                i: ch for ch, i in self.tgt_vocab.items()}
        return self.tgt_inv_vocab.get(idx, '<UNK>')

    def load_data_from_stream(self, stream=stdin, frozen_vocab=False):
        self.samples = []
        self.raw_samples = []
        for line in stream:
            src, tgt = line.rstrip('\n').split('\t')
            src = src.split(' ')
            tgt = tgt.split(' ')
            self.raw_samples.append((src, tgt))
        PAD = Dataset.CONSTANTS['PAD']
        EOS = Dataset.CONSTANTS['EOS']
        UNK = Dataset.CONSTANTS['UNK']
        for src, tgt in self.raw_samples:
            if frozen_vocab is True:
                self.samples.append((
                    [self.src_vocab.get(c, UNK) for c in src],
                    [self.tgt_vocab.get(c, UNK) for c in tgt] + [EOS]
                ))
            else:
                self.samples.append((
                    [self.src_vocab[c] for c in src],
                    [self.tgt_vocab[c] for c in tgt] + [EOS]
                ))

    @staticmethod
    def pad_batch(batch):
        src_len = [len(s[0]) for s in batch]
        tgt_len = [len(s[1]) for s in batch]
        src_maxlen = max(src_len)
        tgt_maxlen = max(tgt_len)
        PAD = Dataset.CONSTANTS['PAD']
        src = [
            [PAD for _ in range(src_maxlen-len(s[0]))] + s[0]
            for s in batch
        ]
        tgt = [
            s[1] + [PAD for _ in range(tgt_maxlen-len(s[1]))]
            for s in batch
        ]
        return src, tgt, src_len, tgt_len

    @staticmethod
    def pad_and_sort_batch(batch):
        src, tgt, src_len, tgt_len = Dataset.pad_batch(batch)
        batch = zip(src, tgt, src_len, tgt_len)
        batch = sorted(batch, key=lambda x: -x[2])
        src, tgt, src_len, tgt_len = zip(*batch)
        return src, tgt, src_len, tgt_len

    def get_random_batch(self, batch_size):
        idx = np.random.choice(range(len(self.samples)), batch_size)
        idx = sorted(idx, key=lambda i: -self.src_seqlen[i])
        src = [self.samples[i][0] for i in idx]
        tgt = [self.samples[i][1] for i in idx]
        src, tgt, src_len, tgt_len = self.pad_and_sort_batch(zip(src, tgt))

        src = Variable(torch.LongTensor(src)).transpose(0, 1)
        tgt = Variable(torch.LongTensor(tgt)).transpose(0, 1)
        tgt_len = Variable(torch.LongTensor(tgt_len))

        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            tgt_len = tgt_len.cuda()

        return src, tgt, src_len, tgt_len

    def batched_iter(self, batch_size):
        batch_count = int(np.ceil(len(self.samples) / batch_size))
        for i in range(batch_count):
            start = i * batch_size
            end = min((i+1) * batch_size, len(self.samples))
            batch = self.samples[start:end]
            src, tgt, src_len, tgt_len = self.pad_and_sort_batch(batch)

            src = Variable(torch.LongTensor(src)).transpose(0, 1)
            tgt = Variable(torch.LongTensor(tgt)).transpose(0, 1)
            tgt_len = Variable(torch.LongTensor(tgt_len))

            if use_cuda:
                src = src.cuda()
                tgt = tgt.cuda()
                tgt_len = tgt_len.cuda()

            yield src, tgt, src_len, tgt_len

    def save_vocabs(self):
        src_fn = os.path.join(self.cfg.experiment_dir, 'src_vocab')
        with open(src_fn, 'w') as f:
            f.write("\n".join(
                "{}\t{}".format(s, t) for s, t in sorted(self.src_vocab.items())
            ))
        tgt_fn = os.path.join(self.cfg.experiment_dir, 'tgt_vocab')
        with open(tgt_fn, 'w') as f:
            f.write("\n".join(
                "{}\t{}".format(s, t) for s, t in sorted(self.tgt_vocab.items())
            ))


class ValidationDataset(Dataset):
    def __init__(self, train_data, stream):
        self.cfg = train_data.cfg
        self.src_vocab = train_data.src_vocab
        self.tgt_vocab = train_data.tgt_vocab
        self.load_data_from_stream(frozen_vocab=True, stream=stream)


class InferenceDataset(Dataset):
    def __init__(self, cfg, stream=None, train_data=None, words=None,
                 spaces=True):
        self.cfg = cfg
        if stream is not None:
            self.load_vocabs()
            self.load_data_from_stream(stream=stream, spaces=spaces)
        elif words is not None:
            self.__copy_attrs(train_data)
            self.load_words(words)

    def __copy_attrs(self, dataset):
        self.src_vocab = dataset.src_vocab
        self.tgt_vocab = dataset.tgt_vocab
        # self.src_maxlen = dataset.src_maxlen

    def load_vocabs(self):
        with open(self.cfg.src_vocab_file) as f:
            self.src_vocab = {}
            for l in f:
                src, tgt = l.rstrip("\n").split("\t")
                self.src_vocab[src] = int(tgt)
        with open(self.cfg.tgt_vocab_file) as f:
            self.tgt_vocab = {}
            for l in f:
                src, tgt = l.rstrip("\n").split("\t")
                self.tgt_vocab[src] = int(tgt)

    def load_data_from_stream(self, stream=stdin, spaces=True):
        if spaces is True:
            samples = [l.rstrip("\n").split("\t")[0].split(" ") for l in stream]
        else:
            samples = [list(l.rstrip("\n").split("\t")[0]) for l in stream]
        self.load_words(samples)

    def load_words(self, words):
        self.raw_samples = words
        PAD = Dataset.CONSTANTS['PAD']
        UNK = Dataset.CONSTANTS['UNK']
        maxlen = max(len(s) for s in self.raw_samples)
        self.samples = [
            [PAD] * (maxlen-len(src)) +
            [self.src_vocab.get(c, UNK) for c in src]
            for src in self.raw_samples
        ]
        self.src_len = [len(s) for s in self.raw_samples]

    def batched_iter(self, batch_size):
        PAD = Dataset.CONSTANTS['PAD']
        batch_count = int(np.ceil(len(self.samples) / batch_size))
        for i in range(batch_count):
            start = i * batch_size
            end = min((i+1) * batch_size, len(self.samples))
            batch = self.samples[start:end]
            batch_len = [len(s) for s in batch]
            maxlen = max(batch_len)
            batch = [
                [PAD] * (maxlen-len(sample)) + sample
                for sample in batch
            ]
            batch = Variable(torch.LongTensor(batch)).transpose(0, 1)
            batch = batch.cuda() if use_cuda else batch
            yield batch, batch_len

    def decode_and_reorganize(self, outputs):
        self.tgt_inv_vocab = {v: k for k, v in self.tgt_vocab.items()}
        for word in outputs:
            word.symbols = [self.tgt_inv_vocab[s] for s in word.idx]
        inv_len_mapping = {v: i for i, v in enumerate(self.len_mapping)}
        decoded = []
        for src_i in range(len(outputs)):
            tgt_i = inv_len_mapping[src_i]
            decoded.append(outputs[tgt_i])
            outputs[tgt_i].input = self.raw_samples[src_i]
        return decoded

    def decode_and_reorganize_beams(self, outputs):
        self.tgt_inv_vocab = {v: k for k, v in self.tgt_vocab.items()}
        EOS = self.CONSTANTS['EOS']
        for words in outputs:
            for word in words[1]:
                try:
                    word.idx = word.idx[:word.idx.index(EOS)]
                except ValueError:
                    pass
                word.symbols = [self.tgt_inv_vocab[s] for s in word.idx]
        return outputs


class DecodedWord(object):
    __slots__ = ('idx', 'prob', 'symbols', 'decoder_hidden', 'input')

    def __init__(self):
        self.idx = []
        self.prob = 1.0
