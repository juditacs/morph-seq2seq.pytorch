#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import logging

import torch

from morph_seq2seq.config import Config
from morph_seq2seq.data import Dataset, ValidationDataset, InferenceDataset
from morph_seq2seq.model import Seq2seqModel, Result


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="YAML config file location")
    return p.parse_args()


class Experiment(object):
    def __init__(self, cfg):
        if isinstance(cfg, str):
            self.cfg = Config.from_yaml(cfg)
        else:
            self.cfg = cfg
        with open(self.cfg.train_file) as f:
            train_data = Dataset(self.cfg, f)
        with open(self.cfg.dev_file) as f:
            val_data = ValidationDataset(train_data, f)
        if hasattr(self.cfg, 'toy_eval'):
            test_data = InferenceDataset(
                self.cfg, train_data=train_data, words=self.cfg.toy_eval)
        else:
            test_data = None
        self.model = Seq2seqModel(train_data, val_data, self.cfg,
                                  toy_data=test_data)
        if use_cuda:
            self.model = self.model.cuda()
        self.train_data = train_data

    def __enter__(self):
        self.result = Result()
        self.model.result = self.result
        self.result.start()
        return self

    def __exit__(self, *args):
        logging.info("Saving experiment to {}".format(
            self.cfg.experiment_dir))
        self.result.stop()
        fn = os.path.join(self.cfg.experiment_dir, 'config.yaml')
        self.cfg.save(fn)
        fn = os.path.join(self.cfg.experiment_dir, 'result.yaml')
        self.result.save(fn)
        self.train_data.save_vocabs()

    def run(self):
        logging.info("Starting training")
        self.model.run_train_schedule()


def main():
    args = parse_args()
    with Experiment(args.config) as e:
        e.run()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    use_cuda = torch.cuda.is_available()
    main()
