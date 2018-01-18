#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import numpy as np
import os
import yaml
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from morph_seq2seq.data import Dataset, InferenceDataset, DecodedWord
from morph_seq2seq.loss import masked_cross_entropy

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, cfg):
        super(self.__class__, self).__init__()
        self.cfg = cfg

        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.embedding = nn.Embedding(cfg.input_size, cfg.src_embedding_size)
        self.__init_cell()

    def __init_cell(self):
        if self.cfg.cell_type == 'LSTM':
            self.cell = nn.LSTM(
                self.cfg.src_embedding_size, self.cfg.hidden_size,
                num_layers=self.cfg.encoder_n_layers,
                bidirectional=True,
                dropout=self.cfg.dropout,
            )
        elif self.cfg.cell_type == 'GRU':
            self.cell = nn.GRU(
                self.cfg.src_embedding_size, self.cfg.hidden_size,
                num_layers=self.cfg.encoder_n_layers,
                bidirectional=True,
                dropout=self.cfg.dropout,
            )

    def forward(self, input, input_seqlen):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
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
        self.n_layers = cfg.decoder_n_layers
        self.embedding_size = cfg.tgt_embedding_size

        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.__init_cell()
        self.concat = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn = Attention('general', self.hidden_size)

    def __init_cell(self):
        if self.cfg.cell_type == 'LSTM':
            self.cell = nn.LSTM(
                self.cfg.tgt_embedding_size, self.cfg.hidden_size,
                num_layers=self.cfg.decoder_n_layers,
                bidirectional=False,
                dropout=self.cfg.dropout,
            )
        elif self.cfg.cell_type == 'GRU':
            self.cell = nn.GRU(
                self.cfg.tgt_embedding_size, self.cfg.hidden_size,
                num_layers=self.cfg.decoder_n_layers,
                bidirectional=False,
                dropout=self.cfg.dropout,
            )

    def forward(self, input_seq, last_hidden, encoder_outputs):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)
        rnn_output, hidden = self.cell(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, attn_weights


class Seq2seqModel(nn.Module):
    def __init__(self, train_data, val_data, cfg, toy_data=None):
        super(self.__class__, self).__init__()
        self.encoder = EncoderRNN(cfg)
        if cfg.attention == 'luong':
            self.decoder = LuongAttentionDecoder(cfg)
        else:
            raise NotImplementedError()
        self.dataset = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.toy_data = toy_data

    def init_optim(self, lr):
        self.enc_opt = getattr(optim, self.cfg.optimizer)(
            self.encoder.parameters(), lr=lr, **self.cfg.optimizer_kwargs)
        self.dec_opt = getattr(optim, self.cfg.optimizer)(
            self.decoder.parameters(), lr=lr, **self.cfg.optimizer_kwargs)

    def run_train_schedule(self):
        self.max_val_loss = 1000
        epoch_offset = 0
        for step in self.cfg.train_schedule:
            logging.info("Running training step {}".format(step))
            self.init_optim(step['lr'])
            for epoch in range(epoch_offset, epoch_offset+step['epochs']):
                loss = 0.0
                self.train(True)
                for ti, batch in enumerate(
                        self.dataset.batched_iter(step['batch_size'])):
                    loss += self.train_batch(batch)
                loss /= (ti + 1)
                # validation
                self.train(False)
                val_loss = 0.0
                for val_i, batch in enumerate(
                        self.val_data.batched_iter(step['batch_size'])):
                    val_loss += self.run_val_batch(batch)
                val_loss /= (val_i + 1)
                logging.info("Epoch {}, train loss {}, val loss {}".format(
                    epoch, loss, val_loss))
                if val_loss < self.max_val_loss:
                    self.max_val_loss = val_loss
                    self.save_model(epoch)
                try:
                    self.result.train_loss.append(loss)
                    self.result.val_loss.append(val_loss)
                except AttributeError:
                    pass
                self.eval_toy()
            epoch_offset += step['epochs']

    def eval_toy(self):
        if self.toy_data is None:
            return
        decoded = self.run_greedy_inference(self.toy_data)
        words = self.toy_data.decode_and_reorganize(decoded)
        for i, word in enumerate(words):
            logging.info("{}\t{}".format(
                "".join(word.input), "".join(word.symbols)))

    def save_model(self, epoch):
        save_path = os.path.join(
            self.cfg.experiment_dir,
            "model.epoch_{}".format("{0:04d}".format(epoch)))
        logging.info("Saving model to {}".format(save_path))
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
            if np.random.random() <= self.cfg.teacher_forcing_ratio:
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
        return loss.data[0]

    def init_decoder_hidden(self, encoder_hidden):
        if self.cfg.cell_type == 'LSTM':
            decoder_hidden = tuple(e[:self.decoder.n_layers]
                                   for e in encoder_hidden)
        else:
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        return decoder_hidden

    def run_val_batch(self, batch):
        return self.train_batch(batch, do_train=False)

    def run_greedy_inference(self, test_data):
        return self.run_inference(test_data, 'greedy')

    def run_beam_search_inference(self, test_data, beam_width):
        return self.run_inference(test_data, 'beam_search', beam_width)

    def run_inference(self, test_data, mode='greedy', beam_width=3):
        assert isinstance(test_data, InferenceDataset)
        if self.cfg.eval_batch_size is None:
            batch_size = self.cfg.train_schedule[0]['batch_size']
        else:
            batch_size = self.cfg.eval_batch_size
        all_output = []
        for bi, (src, src_len) in enumerate(test_data.batched_iter(batch_size)):
            logging.info("Batch {}, samples {}".format(bi+1, bi*batch_size))
            all_encoder_outputs, encoder_hidden = self.encoder(src, src_len)

            if isinstance(encoder_hidden, tuple):
                decoder_hidden = tuple(e[:self.cfg.decoder_n_layers]
                                       for e in encoder_hidden)
            else:
                decoder_hidden = encoder_hidden[:self.cfg.decoder_n_layers]
            all_decoder_hidden = decoder_hidden
            for si in range(len(src_len)):
                if isinstance(all_decoder_hidden, tuple):
                    decoder_hidden = tuple(
                        a[:, si, :].unsqueeze(1).contiguous()
                        for a in all_decoder_hidden)
                else:
                    decoder_hidden = all_decoder_hidden[:, si, :].unsqueeze(
                        1).contiguous()
                encoder_outputs = all_encoder_outputs[:, si, :].unsqueeze(1)
                maxlen = max(src_len) * 3  # arbitrary
                if mode == 'greedy':
                    all_output.append(self.__decode_sample_greedy(
                        encoder_outputs,
                        decoder_hidden,
                        dataset=test_data,
                        maxlen=maxlen,
                    ))
                elif mode == 'beam_search':
                    decoder = BeamSearchDecoder(
                        self.decoder, beam_width, encoder_outputs,
                        decoder_hidden, maxlen)
                    while decoder.is_finished() is False:
                        decoder.forward()
                    all_output.append(decoder.get_finished_candidates())
        return all_output

    def __decode_sample_greedy(self, encoder_outputs, decoder_hidden, dataset,
                               maxlen):
        decoder_input = Variable(torch.LongTensor(
            [dataset.CONSTANTS['SOS']]))
        if use_cuda:
            decoder_input = decoder_input.cuda()
        word = DecodedWord()
        for i in range(maxlen):
            decoder_output, decoder_hidden, da = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_output = self.softmax(decoder_output)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == dataset.CONSTANTS['EOS']:
                break
            word.idx.append(ni)
            word.prob *= topv[0][0]
            decoder_input = Variable(torch.LongTensor([ni]))
            if use_cuda:
                decoder_input = decoder_input.cuda()
        return word


class Beam(object):
    @classmethod
    def from_single_idx(cls, output, idx, hidden):
        beam = cls()
        beam.output = output
        beam.probs = [output.data[0, idx]]
        beam.idx = [idx]
        beam.hidden = hidden
        return beam

    @classmethod
    def from_existing(cls, source, output, idx, hidden):
        beam = cls()
        beam.output = output
        beam.probs = source.probs.copy()
        beam.probs.append(output.data[0, idx])
        beam.idx = source.idx.copy()
        beam.idx.append(idx)
        beam.hidden = hidden
        return beam

    def decode(self, data):
        try:
            eos = data.CONSTANTS['EOS']
            self.idx = self.idx[:self.idx.index(eos)]
        except ValueError:
            pass
        rev = [data.tgt_reverse_lookup(s) for s in self.idx]
        return "".join(rev)

    def is_finished(self):
        return len(self.idx) > 0 and \
            self.idx[-1] == Dataset.CONSTANTS['EOS']

    def __len__(self):
        return len(self.idx)

    @property
    def prob(self):
        p = 1.0
        for o in self.probs:
            p *= o
        return p


class BeamSearchDecoder(nn.Module):
    def __init__(self, decoder, width, encoder_outputs, encoder_hidden,
                 max_iter):
        super(self.__class__, self).__init__()
        self.decoder = decoder
        self.width = width
        self.encoder_hidden = encoder_hidden
        self.encoder_outputs = encoder_outputs
        self.decoder_outputs = []
        self.softmax = nn.Softmax(dim=1)
        self.init_candidates()
        self.max_iter = max_iter
        self.finished_candidates = []

    def init_candidates(self):
        self.candidates = []
        decoder_input = Variable(torch.LongTensor([Dataset.CONSTANTS['SOS']]))
        if use_cuda:
            decoder_input = decoder_input.cuda()
        if isinstance(self.encoder_hidden, tuple):
            decoder_hidden = tuple(e[:self.decoder.n_layers]
                                   for e in self.encoder_hidden)
        else:
            decoder_hidden = self.encoder_hidden[:self.decoder.n_layers]
        output, hidden, _ = self.decoder(decoder_input, decoder_hidden,
                                         self.encoder_outputs)
        output = self.softmax(output)
        top_out, top_idx = output.data.topk(self.width)
        for i in range(top_out.size()[1]):
            self.candidates.append(Beam.from_single_idx(
                output=output, idx=top_idx[0, i], hidden=hidden))

    def is_finished(self):
        if self.max_iter < 0:
            return True
        return len(self.candidates) == self.width and \
            all(c.is_finished() for c in self.candidates)

    def forward(self):
        self.max_iter -= 1
        if self.max_iter < 0:
            return
        new_candidates = []
        for c in self.candidates:
            if c.is_finished():
                self.finished_candidates.append(c)
                continue
            decoder_input = Variable(torch.LongTensor([c.idx[-1]]))
            if use_cuda:
                decoder_input = decoder_input.cuda()
            output, hidden, _ = self.decoder(
                decoder_input, c.hidden, self.encoder_outputs)
            output = self.softmax(output)
            top_out, top_idx = output.data.topk(self.width)
            for i in range(top_out.size()[1]):
                new_candidates.append(
                    Beam.from_existing(source=c, output=output,
                                       idx=top_idx[0, i], hidden=hidden))
        self.candidates = sorted(
            new_candidates, key=lambda x: -x.prob)[:self.width]

    def get_finished_candidates(self):
        top = sorted(self.candidates + self.finished_candidates,
                     key=lambda x: -x.prob)[:self.width]
        for t in top:
            delattr(t, 'hidden')
            delattr(t, 'output')
        return top


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
