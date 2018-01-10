#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import yaml
import re


class ConfigError(ValueError):
    pass


class Config(object):
    # path variables support environment variable
    # ${MYVAR} will be manually expanded
    path_variables = (
        'train_file', 'dev_file', 'experiment_dir'
    )
    __slots__ = (
        'share_vocab', 'src_embedding_size', 'tgt_embedding_size', 'batch_size',
        'encoder_n_layers', 'decoder_n_layers', 'dropout_prob',
        'cell_type', 'hidden_size', 'input_size', 'output_size',
        'attention', 'optimizer', 'optimizer_kwargs',
        'train_schedule', 'generate_empty_subdir', 'teacher_forcing_ratio',
        'src_vocab_file', 'tgt_vocab_file', 'derive_vocab',
        'toy_eval', 'eval_batch_size',
    ) + path_variables

    defaults = {
        'share_vocab': False,
        'derive_vocab': False,
        'attention': 'luong',
        'optimizer': 'SGD',
        'optimizer_kwargs': {},
        'teacher_forcing_ratio': 0.8,
        'src_vocab_file': None,
        'tgt_vocab_file': None,
    }

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            params = yaml.load(f)
        return cls(**params)

    @classmethod
    def from_config_dir(cls, config_dir):
        """Find config.yaml in config_dir and load.
        Used for inference
        """
        yaml_fn = os.path.join(config_dir, 'config.yaml')
        cfg = cls.from_yaml(yaml_fn)
        cfg.config_dir = config_dir
        return cfg

    def __init__(self, **kwargs):
        for param, val in self.defaults.items():
            setattr(self, param, val)
        for param, val in kwargs.items():
            setattr(self, param, val)
        self.expand_variables()
        self.derive_params()
        self.validate_params()

    def expand_variables(self):
        var_re = re.compile(r'\$\{([^}]+)\}')
        for p in Config.path_variables:
            v = getattr(self, p)
            v_cpy = v
            for m in var_re.finditer(v):
                key = m.group(1)
                v_cpy = v_cpy.replace(m.group(0), os.environ[key])
            setattr(self, p, v_cpy)

    def derive_params(self):
        if self.generate_empty_subdir is True:
            i = 0
            fmt = '{0:04d}'
            while os.path.exists(os.path.join(self.experiment_dir,
                                              fmt.format(i))):
                i += 1
            self.experiment_dir = os.path.join(
                self.experiment_dir, fmt.format(i))
            os.makedirs(self.experiment_dir)
        if self.src_vocab_file is None:
            self.src_vocab_file = os.path.join(self.experiment_dir, 'src_vocab')
            self.tgt_vocab_file = os.path.join(self.experiment_dir, 'tgt_vocab')

    def validate_params(self):
        if self.attention not in ('luong', 'bahdanau'):
            raise ConfigError("Attention type must be luong or bahdanau")
        if self.cell_type not in ('LSTM', 'GRU'):
            raise ConfigError("Cell type must be LSTM or GRU")
        if self.teacher_forcing_ratio < 0 or self.teacher_forcing_ratio > 1:
            raise ConfigError("Teacher forcing ratio must be between 0 and 1")
        if self.derive_vocab is True:
            if not os.path.exists(self.src_vocab_file) or \
                    not os.path.exists(self.tgt_vocab_file):
                raise ConfigError("Src and tgt vocab files must exist if "
                                  "derive_vocab is True")

    def save(self, fn):
        d = {k: getattr(self, k, None) for k in self.__slots__}
        with open(fn, 'w') as f:
            yaml.dump(d, f)


class InferenceConfig(Config):
    def __init__(self, **kwargs):
        kwargs['generate_empty_subdir'] = False
        super(self.__class__, self).__init__(**kwargs)
