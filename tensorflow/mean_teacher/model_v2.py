# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"Mean teacher model"

import logging
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean
import pickle

from . import nn
from . import weight_norm as wn
from .framework import ema_variable_scope, name_variable_scope, assert_shape, HyperparamVariables
from . import string_utils


LOG = logging.getLogger('main')


class Model:
    DEFAULT_HYPERPARAMS = {
        # Consistency hyperparameters
        'ema_consistency': True,
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'consistency_trust': 0.0,
        'num_logits': 1, # Either 1 or 2
        'logit_distance_cost': 0.0, # Matters only with 2 outputs

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,
    }

    #pylint: disable=too-many-instance-attributes
    def __init__(self, run_context=None,inputsize=32):
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, inputsize, inputsize, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')
            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')

        (
            (self.class_logits_1, self.cons_logits_1),
            (self.class_logits_2, self.cons_logits_2),
            (self.class_logits_ema, self.cons_logits_ema)
        ) = inference(
            self.images,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'],
            translate=self.hyper['translate'],
            num_logits=self.hyper['num_logits'],inputsize=inputsize)

        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels)
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels)

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels)
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels)

            self.class_label_1=tf.nn.softmax(self.class_logits_1)
            self.class_label_2=tf.nn.softmax(self.class_logits_2)
            self.class_label_ema=tf.nn.softmax(self.class_logits_ema)

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_pi, self.cons_costs_pi = consistency_costs(
                self.cons_logits_1, self.class_logits_2, self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'])
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.cons_logits_1, self.class_logits_ema, self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'])


            def l2_norms(matrix):
                l2s = tf.reduce_sum(matrix ** 2, axis=1)
                mean_l2 = tf.reduce_mean(l2s)
                return mean_l2, l2s

            self.mean_res_l2_1, self.res_l2s_1 = l2_norms(self.class_logits_1 - self.cons_logits_1)
            self.mean_res_l2_ema, self.res_l2s_ema = l2_norms(self.class_logits_ema - self.cons_logits_ema)
            self.res_costs_1 = self.hyper['logit_distance_cost'] * self.res_l2s_1
            self.mean_res_cost_1 = tf.reduce_mean(self.res_costs_1)
            self.res_costs_ema = self.hyper['logit_distance_cost'] * self.res_l2s_ema
            self.mean_res_cost_ema = tf.reduce_mean(self.res_costs_ema)

            self.mean_total_cost_pi, self.total_costs_pi = total_costs(
                self.class_costs_1, self.cons_costs_pi, self.res_costs_1)
            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt, self.res_costs_1)
            assert_shape(self.total_costs_pi, [3])
            assert_shape(self.total_costs_mt, [3])

            self.cost_to_be_minimized = tf.cond(self.hyper['ema_consistency'],
                                                lambda: self.mean_total_cost_mt,
                                                lambda: self.mean_total_cost_pi)

        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])

        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/pi": self.mean_cons_cost_pi,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/res_cost/1": self.mean_res_cost_1,
            "train/res_cost/ema": self.mean_res_cost_ema,
            "train/total_cost/pi": self.mean_total_cost_pi,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
                "eval/res_cost/1": streaming_mean(self.res_costs_1),
                "eval/res_cost/ema": streaming_mean(self.res_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["eval/error/ema", "error/1", "class_cost/1", "cons_cost/mt"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()

        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.session = tf.Session()
        self.run(self.init_init_op)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn):
        self.run(self.train_init_op, self.feed_dict(next(training_batches)))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        for batch in training_batches:
            results, _ = self.run([self.training_metrics, self.train_step_op],
                                  self.feed_dict(batch))
            step_control = self.get_training_control()
#            self.training_log.record(step_control['step'], {**results, **step_control})
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        count=0
        class_label_1=[]
        class_label_2=[]
        class_label_ema=[]
        real_labels=[]
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))

            temp_label_1,temp_label_2,temp_ema=self.run([self.class_label_1, self.class_label_2,self.class_label_ema],
                     feed_dict=self.feed_dict(batch, is_training=False))

            class_label_1.extend(temp_label_1)
            class_label_2.extend(temp_label_2)
            class_label_ema.extend(temp_ema)
            real_labels.extend(batch['y'])

           # print(str(count)+'\n')
            #count+=1
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
#        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

        fwo=open('logits_result','wb')
        pickle.dump(class_label_1,fwo)
        pickle.dump(class_label_2,fwo)
        pickle.dump(class_label_ema,fwo)
        pickle.dump(real_labels,fwo)
        fwo.close()

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()

    def load_checkpoint(self,checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        self.saver.restore(self.session,checkpoint_path)

Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")


def inference(inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
              normalize_input, flip_horizontally, translate, num_logits,inputsize=32):
    tower_args = dict(inputs=inputs,
                      is_training=is_training,
                      input_noise=input_noise,
                      normalize_input=normalize_input,
                      flip_horizontally=flip_horizontally,
                      translate=translate,
                      num_logits=num_logits)

    with tf.variable_scope("initialization") as var_scope:
        _ = tower(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True,inputsize=inputsize)
    with name_variable_scope("primary", var_scope, reuse=True) as (name_scope, _):
        class_logits_1, cons_logits_1 = tower(**tower_args, dropout_probability=student_dropout_probability, name=name_scope,inputsize=inputsize)
    with name_variable_scope("secondary", var_scope, reuse=True) as (name_scope, _):
        class_logits_2, cons_logits_2 = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope,inputsize=inputsize)
    with ema_variable_scope("ema", var_scope, decay=ema_decay):
        class_logits_ema, cons_logits_ema = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope,inputsize=inputsize)
        class_logits_ema, cons_logits_ema = tf.stop_gradient(class_logits_ema), tf.stop_gradient(cons_logits_ema)
    return (class_logits_1, cons_logits_1), (class_logits_2, cons_logits_2), (class_logits_ema, cons_logits_ema)


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          num_logits,
          is_initialization=False,
          name=None,inputsize=32):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        min_depth = 16,
        depth_multiplier = 1.0

        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):
            #pylint: disable=no-value-for-parameter
            net = inputs
            assert_shape(net, [None, inputsize, inputsize, 3])

            net = tf.cond(normalize_input,
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)
            assert_shape(net, [None, inputsize, inputsize, 3])

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = tf.cond(translate,
                          lambda: nn.random_translate(net, scale=2, name='random_translate'),
                          lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            # 299 x 299 x 3   should be checked for 128*128*3 for reduce memeory
            end_point = 'Conv2d_1a_3x3'
            net = wn.conv2d(inputs, depth(32), [3, 3], stride=[2,2], scope=end_point)

            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            net = wn.conv2d(net, depth(32), [3, 3], scope=end_point)
            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            net = wn.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)

            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)

            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            net = wn.conv2d(net, depth(80), [1, 1], scope=end_point)

            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            net = wn.conv2d(net, depth(192), [3, 3], scope=end_point)

            # 71 x 71 x 192.
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            # 35 x 35 x 192.

            # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = wn.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(branch_3, depth(32), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(64), [1, 1],
                                           scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = wn.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(branch_3, depth(64), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = wn.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(branch_3, depth(64), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(384), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_1 = wn.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(128), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = wn.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(128), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(128), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = wn.conv2d(branch_2, depth(128), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(160), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = wn.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(160), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = wn.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(160), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = wn.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(160), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = wn.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(192), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = wn.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(branch_2, depth(192), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = wn.conv2d(branch_2, depth(192), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = wn.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = wn.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = wn.conv2d(branch_1, depth(192), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = wn.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                    branch_1 = wn.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        wn.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        wn.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        wn.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        wn.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = wn.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = wn.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        wn.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        wn.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = wn.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = wn.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        wn.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        wn.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = wn.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            net = tf.reduce_mean(net,axis=[1,2])
            assert_shape(net, [None, 2048])

            primary_logits = wn.fully_connected(net, 2, init=is_initialization)
            secondary_logits = wn.fully_connected(net, 2, init=is_initialization)

            with tf.control_dependencies([tf.assert_greater_equal(num_logits, 1),
                                          tf.assert_less_equal(num_logits, 2)]):
                secondary_logits = tf.case([
                    (tf.equal(num_logits, 1), lambda: primary_logits),
                    (tf.equal(num_logits, 2), lambda: secondary_logits),
                ], exclusive=True, default=lambda: primary_logits)

            assert_shape(primary_logits, [None, 2])
            assert_shape(secondary_logits, [None, 2])
            return primary_logits, secondary_logits


def errors(logits, labels, name=None):
    """Compute error mean and whether each unlabeled example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean error over unlabeled examples.
    Mean error is NaN if there are no unlabeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs(logits1, logits2, cons_coefficient, mask, consistency_trust, name=None):
    """Takes a softmax of the logits and returns their distance as described below

    Consistency_trust determines the distance metric to use
    - trust=0: MSE
    - 0 < trust < 1: a scaled KL-divergence but both sides mixtured with
      a uniform distribution with given trust used as the mixture weight
    - trust=1: scaled KL-divergence

    When trust > 0, the cost is scaled to make the gradients
    the same size as MSE when trust -> 0. The scaling factor used is
    2 * (1 - 1/num_classes) / num_classes**2 / consistency_trust**2 .
    To have consistency match the strength of classification, use
    consistency coefficient = num_classes**2 / (1 - 1/num_classes) / 2
    which is 55.5555... when num_classes=10.

    Two potential stumbling blokcs:
    - When trust=0, this gives gradients to both logits, but when trust > 0
      this gives gradients only towards the first logit.
      So do not use trust > 0 with the Pi model.
    - Numerics may be unstable when 0 < trust < 1.
    """

    with tf.name_scope(name, "consistency_costs") as scope:
        num_classes = 2
        assert_shape(logits1, [None, num_classes])
        assert_shape(logits2, [None, num_classes])
        assert_shape(cons_coefficient, [])
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        kl_cost_multiplier = 2 * (1 - 1/num_classes) / num_classes**2 / consistency_trust**2

        def pure_mse():
            costs = tf.reduce_mean((softmax1 - softmax2) ** 2, -1)
            return costs

        def pure_kl():
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=softmax2)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=softmax2)
            costs = cross_entropy - entropy
            costs = costs * kl_cost_multiplier
            return costs

        def mixture_kl():
            with tf.control_dependencies([tf.assert_greater(consistency_trust, 0.0),
                                          tf.assert_less(consistency_trust, 1.0)]):
                uniform = tf.constant(1 / num_classes, shape=[num_classes])
                mixed_softmax1 = consistency_trust * softmax1 + (1 - consistency_trust) * uniform
                mixed_softmax2 = consistency_trust * softmax2 + (1 - consistency_trust) * uniform
                costs = tf.reduce_sum(mixed_softmax2 * tf.log(mixed_softmax2 / mixed_softmax1), axis=1)
                costs = costs * kl_cost_multiplier
                return costs

        costs = tf.case([
            (tf.equal(consistency_trust, 0.0), pure_mse),
            (tf.equal(consistency_trust, 1.0), pure_kl)
        ], default=mixture_kl)

        costs = costs * tf.to_float(mask) * cons_coefficient
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs


def total_costs(*all_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        for cost in all_costs:
            assert_shape(cost, [None])
        costs = tf.reduce_sum(all_costs, axis=1)
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs
