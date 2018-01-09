#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os

import torch

from morph_seq2seq.config import Config
from morph_seq2seq.data import Dataset, ValidationDataset
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
        self.model = Seq2seqModel(train_data, val_data, self.cfg)
        if use_cuda:
            self.model = self.model.cuda()

    def __enter__(self):
        self.result = Result()
        self.model.result = self.result
        self.result.start()
        return self

    def __exit__(self, *args):
        self.result.stop()
        fn = os.path.join(self.cfg.experiment_dir, 'config.yaml')
        self.cfg.save(fn)
        fn = os.path.join(self.cfg.experiment_dir, 'result.yaml')
        self.result.save(fn)

    def run(self):
        self.model.run_train_schedule()


def main():
    args = parse_args()
    with Experiment(args.config) as e:
        e.run()
        return
    cfg = Config.from_yaml(args.config)
    with open(cfg.train_file) as f:
        train_data = Dataset(cfg, f)
    with open(cfg.dev_file) as f:
        val_data = ValidationDataset(train_data=train_data, stream=f)
    model = Seq2seqModel(train_data, val_data, cfg)
    with Result(cfg) as result:
        model.result = result
        if use_cuda:
            model = model.cuda()
        model.run_train_schedule()


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    main()
