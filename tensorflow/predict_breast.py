# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Train ConvNet Mean Teacher on SVHN training set and evaluate against a validation set

This runner converges quickly to a fairly good accuracy.
On the other hand, the runner experiments/svhn_final_eval.py
contains the hyperparameters used in the paper, and converges
much more slowly but possibly to a slightly better accuracy.
"""

import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets import breast
from mean_teacher.model import Model
from mean_teacher import minibatching


logging.getLogger('main').setLevel(logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 'all'
    n_extra_unlabeled = 4385

    model = Model(RunContext(__file__, 0),inputsize=128)
    model['rampdown_length'] = 0
    model['rampup_length'] = 5000
    model['training_length'] = 40000
    model['max_consistency_cost'] = 50.0

    #turn off data argumentation
    checkpoint_path = 'results/train_breast/2020-09-25_21:31:02/0/transient'
    model.load_checkpoint(checkpoint_path)

    data = breast.Breast(data_seed, n_labeled, n_extra_unlabeled,test_phase=True)
    training_batches = minibatching.training_batches(data.training, n_labeled_per_batch=32,batch_size=64)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation)
    model.evaluate(evaluation_batches_fn)


if __name__ == "__main__":
    run()
