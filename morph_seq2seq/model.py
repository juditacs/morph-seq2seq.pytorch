#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin
from collections import defaultdict
import numpy as np
import os
import yaml
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from morph_seq2seq.data import Dataset
from morph_seq2seq.loss import masked_cross_entropy

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, cfg):
        super(self.__class__, self).__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.input_size, cfg.src_embedding_size)
        self.__init_cell()

    def __init_cell(self):
        if self.cfg.cell_type == 'LSTM':
            self.cell = nn.LSTM(
                self.cfg.src_embedding_size, self.cfg.hidden_size,
                num_layers=self.cfg.num_layers,
                bidirectional=True
            )
        elif self.cfg.cell_type == 'GRU':
            self.cell = nn.GRU(
                self.cfg.src_embedding_size, self.cfg.hidden_size,
                num_layers=self.cfg.num_layers,
                bidirectional=True
            )

    def forward(self, input, input_seqlen):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_seqlen)
        outputs, hidden = self.cell(packed)
        outputs, ol = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.cfg.hidden_size] + \
            outputs[:, :, self.cfg.hidden_size:]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(self.__class__, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs).transpose(0, 1)
        e = energy.bmm(hidden.transpose(0, 1).transpose(1, 2))
        energies = e.squeeze(2)
        return F.softmax(energies, 1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            return hidden.dot(encoder_output)
        if self.method == 'general':
            energy = self.attn(encoder_output)
            return hidden.dot(energy)
        elif self.method == 'concat':
            energy = torch.cat((hidden, encoder_output), 0)
            energy = self.attn(energy.unsqueeze(0))
            energy = self.v.dot(energy)
            return energy


class LuongAttentionDecoder(nn.Module):
    def __init__(self, cfg):
        super(self.__class__, self).__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.output_size = cfg.output_size
        self.n_layers = cfg.num_layers
        self.embedding_size = cfg.tgt_embedding_size

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.gru = nn.LSTM(self.embedding_size, self.hidden_size,
                           num_layers=self.n_layers)
        self.concat = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn = Attention('general', self.hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = embedded.view(1, batch_size, self.embedding_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, attn_weights


class Seq2seqModel(nn.Module):
    def __init__(self, train_data, val_data, cfg):
        super(self.__class__, self).__init__()
        self.encoder = EncoderRNN(cfg)
        if cfg.attention == 'luong':
            self.decoder = LuongAttentionDecoder(cfg)
        else:
            raise NotImplementedError()
        self.dataset = train_data
        self.val_data = val_data
        self.cfg = cfg

    def init_optim(self, lr):
        self.enc_opt = getattr(optim, self.cfg.optimizer)(
            self.encoder.parameters(), lr=lr, **self.cfg.optimizer_kwargs)
        self.dec_opt = getattr(optim, self.cfg.optimizer)(
            self.decoder.parameters(), lr=lr, **self.cfg.optimizer_kwargs)

    def run_train_schedule(self):
        self.max_val_loss = 1000
        epoch_offset = 0
        for step in self.cfg.train_schedule:
            self.init_optim(step['lr'])
            for epoch in range(epoch_offset, epoch_offset+step['epochs']):
                self.train(True)
                for ti, batch in enumerate(
                    self.dataset.batched_iter(step['batch_size'])):
                    loss = self.train_batch(batch)
                loss /= (ti + 1)
                # validation
                self.train(False)
                val_loss = 0.0
                for val_i, batch in enumerate(
                    self.val_data.batched_iter(step['batch_size'])):
                    val_loss += self.run_val_batch(batch)
                val_loss /= (val_i + 1)
                if val_loss < self.max_val_loss:
                    self.max_val_loss = val_loss
                    self.save_model(epoch)
                try:
                    self.result.train_loss.append(loss)
                    self.result.val_loss.append(val_loss)
                except AttributeError:
                    pass

    def save_model(self, epoch):
        save_path = os.path.join(
            self.cfg.experiment_dir,
            "model.epoch_{}".format("{0:04d}".format(epoch)))
        torch.save(self.state_dict(), save_path)

    def train_batch(self, batch, do_train=True):
        src, tgt, src_len, tgt_len = batch
        batch_size = src.size(1)
        encoder_outputs, encoder_hidden = self.encoder(src, src_len)

        decoder_hidden = self.init_decoder_hidden(encoder_hidden)
        decoder_input = Variable(torch.LongTensor(
            [Dataset.CONSTANTS['SOS']] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        all_decoder_outputs = Variable(torch.zeros((
            tgt.size(0), batch_size, len(self.dataset.tgt_vocab))))
        if use_cuda:
            all_decoder_outputs = all_decoder_outputs.cuda()

        for t in range(tgt.size(0)):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output
            if np.random.random() < self.cfg.teacher_forcing_ratio:
                decoder_input = tgt[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            tgt.transpose(0, 1).contiguous(),
            tgt_len
        )
        if do_train:
            loss.backward()
            self.enc_opt.step()
            self.dec_opt.step()
        return loss.data[0] / self.dataset.tgt_maxlen

    def init_decoder_hidden(self, encoder_hidden):
        if self.cfg.cell_type == 'LSTM':
            decoder_hidden = tuple(e[:self.decoder.n_layers]
                                   for e in encoder_hidden)
        else:
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        return decoder_hidden

    def run_val_batch(self, batch):
        return self.train_batch(batch, do_train=False)

    def run_greedy_inference(self, samples):
        data = InferenceDataset(samples)


class Result(object):
    __slots__ = ('train_loss', 'val_loss', 'start_time', 'running_time')

    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def start(self):
        self.start_time = datetime.now()

    def stop(self, *args):
        self.running_time = (datetime.now() - self.start_time).total_seconds()

    def save(self, fn):
        d = {k: getattr(self, k) for k in self.__slots__}
        with open(fn, 'w') as f:
            yaml.dump(d, f)
