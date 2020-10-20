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

#logging.basicConfig(level=logging.INFO)
logging.getLogger('main').setLevel(logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 'all'
    n_extra_unlabeled = 4385

    model = Model(RunContext(__file__, 0),inputsize=128)
    model['rampdown_length'] = 40000
    model['rampup_length'] = 30000
    model['training_length'] = 100000
    model['max_consistency_cost'] = 50.0

    #turn on or off data argumentation
    # model['translate'] = False
    # model['flip_horizontally'] = False

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    data = breast.Breast(data_seed, n_labeled, n_extra_unlabeled,test_phase=False)
    training_batches = minibatching.training_batches(data.training, n_labeled_per_batch=64,batch_size=128)  #64 label and 64 unlabel in batch
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()
