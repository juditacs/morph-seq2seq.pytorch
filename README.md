# morph-seq2seq.pytorch
My working codebase for seq2seq experiments using PyTorch

I am a beginner in PyTorch. If you have any comments or suggestions please file an issue.

## Installing

Clone and install the package via pip:

    git clone git@github.com:juditacs/morph-seq2seq.pytorch.git
    cd morph-seq2seq.pytorch
    pip install -e .

## Training and test file formats

Training and development files are expected to contain one sample-per-line with the input and output separated by TAB. Symbols are separated by spaces. For example a one sentence English-French parallel corpus would look like this (tokenization may differ):

~~~
I am hungry <TAB> J' ai faim
~~~

Most of my experiments are character-level.
For example the training corpus for Hungarian instrumental case looks like this:

~~~
a l m a <TAB> a l m á v a l
k ö r t e <TAB> k ö r t é v e l
v i r á g <TAB> v i r á g g a l
~~~

In this case a single character is a symbol.

Test files are very similar except only the first column is used (if there are
more, the rest are ignored).

## Training

An experiment is described in a YAML configuration file. A toy example is available at `config/toy.yaml`.

Variable such as `${VAR}` are expanded using environment variables.

**WARNING** this is done manually since YAML does not support external variables.
My implementation may easily be exploited, do not run it as a web service or the like.

In the toy example only one such variable is used, you can set it with:

    mkdir -p experiments/toy
    export EXP_DIR=experiments

then you can run the experiment:

    python morph_seq2seq/train.py --config config/toy.yaml

You should see a bunch of log messages:

* the train and validation loss printed after each epoch,
* the model saved after an epoch if the validation loss decreased to a file
  called `model.epoch_NNNN`, where `NNNN` is the epoch number,
* the current output to the toy evaluation set (listed in the variable `toy_eval` in
  the configuration file). I use this to make sure that the model is not
  complete garbage (the toy model will be garbage).

The experiment can be stopped any time with Ctrl+C. The best model will have
already been saved and other files such as result statistics are also saved upon exiting.

## Experiment directory

By default an empty subdirectory is created under `experiment_dir` with a
4-digit name and everything related to the experiment is saved into this
directory.

An experiment directory contains:

1. `config.yaml`: the final full configuration is saved here.
2. `result.yaml`: contains the train and val loss in each epoch and the experiment's timestamp and running time.
3. `src_vocab` and `tgt_vocab`: the source and target language vocabularies.
4. `model.epoch_N` and similar: model parameters after epoch N if the
   validation loss decreased compared to the current minimum.

## Continue training a model

You can continue training a model with:

    python morph_seq2seq/train.py --config config/toy.yaml --load-model
    experiments/toy/0000/model.epoch_0012

but it is considered a new experiment (i.e. a new directory is created).

## Inference

Inference takes and experiment directory as its argument and the last saved
model is used:

    python morph_seq2seq/inference.py --experiment-dir experiments/toy/0000 --test-file data/toy

The default is greedy mode, beam search can be used as:

    python morph_seq2seq/inference.py --experiment-dir experiments/toy/0000 --test-file data/toy --mode beam_search --beam-width 2

The sequence probabilities can be printed as well (in both modes):

    python morph_seq2seq/inference.py --experiment-dir experiments/toy/0000 --test-file data/toy --mode beam_search --beam-width 2 --print-probabilities

