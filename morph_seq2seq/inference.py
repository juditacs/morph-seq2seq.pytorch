#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
from sys import stdin
import logging

import torch

from morph_seq2seq.data import InferenceDataset
from morph_seq2seq.config import InferenceConfig
from morph_seq2seq.model import Seq2seqModel


def parse_args():
    p = ArgumentParser()
    p.add_argument("-e", "--experiment-dir", type=str,
                   help="Experiment directory")
    p.add_argument("-m", "--mode", choices=['greedy', 'beam_search'],
                   default='greedy')
    p.add_argument("-b", "--beam-width", type=int, default=3,
                   help="Beam width. Only used in beam search mode")
    p.add_argument("-t", "--test-file", type=str, default=None,
                   help="Test file location")
    p.add_argument("--print-probabilities", action="store_true",
                   default=False,
                   help="Print the probability of each output sequence")
    return p.parse_args()


class Inference(object):
    def __init__(self, exp_dir, test_file_fn, mode='greedy', beam_width=None):
        self.exp_dir = exp_dir
        cfg = os.path.join(exp_dir, 'config.yaml')
        self.cfg = InferenceConfig.from_yaml(cfg)
        if test_file_fn is not None:
            with open(test_file_fn) as f:
                self.test_data = InferenceDataset(self.cfg, f)
        else:
            self.test_data = InferenceDataset(self.cfg, stdin, spaces=False)
        self.model = Seq2seqModel(cfg=self.cfg, train_data=None, val_data=None)
        self.model = self.model.cuda() if use_cuda else self.model
        self.model.train(False)
        model_fn = self.find_last_model()
        self.model.load_state_dict(torch.load(model_fn))
        self.mode = mode
        self.beam_width = beam_width

    def find_last_model(self):
        saves = filter(lambda f: f.startswith(
            'model.epoch_'), os.listdir(self.exp_dir))
        last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
        return os.path.join(self.exp_dir, last_epoch)

    def run_inference(self):
        if self.mode == 'greedy':
            decoded = self.model.run_greedy_inference(self.test_data)
            words = self.test_data.decode_and_reorganize(decoded)
            return words
        if self.mode == 'beam_search':
            decoded = self.model.run_beam_search_inference(
                self.test_data, self.beam_width)
            words = self.test_data.decode_and_reorganize_beams(decoded)
            return words


def main():
    args = parse_args()
    if args.mode == 'greedy':
        inf_model = Inference(args.experiment_dir, args.test_file, args.mode)
        words = inf_model.run_inference()
        for word in words:
            if args.print_probabilities:
                print("{}\t{}\t{}".format("".join(word.input),
                                          "".join(word.symbols), word.prob))
            else:
                print("{}\t{}".format("".join(word.input),
                                      "".join(word.symbols)))

    elif args.mode == 'beam_search':
        inf_model = Inference(args.experiment_dir, args.test_file, args.mode,
                              args.beam_width)
        words = inf_model.run_inference()
        for word in words:
            out = ["".join(word[0])]
            for beam in word[1]:
                out.append("".join(beam.symbols))
                if args.print_probabilities:
                    out.append(beam.prob)
            print("\t".join(map(str, out)))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    use_cuda = torch.cuda.is_available()
    main()
