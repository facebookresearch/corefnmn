"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script to train MNIST Dialog model using supervised learning.

Trains mnist dialog model that performs explicit visual coreference resolution
using neural module networks. Additional details are in the paper:
  Visual Coreference Resolution in Visual Dialog using Neural Module Networks
  Satwik Kottur, José M. F. Moura, Devi Parikh, Dhruv Batra, Marcus Rohrbach
  European Conference on Computer Vision (ECCV), 2018
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
import time
from tqdm import tqdm as progressbar
import numpy as np
import tensorflow as tf

from exp_mnist import options

# read command line options
args = options.read_command_line()

# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                           allow_soft_placement=False,
                           log_device_placement=False)
sess = tf.Session(config=tf_config)

from models_mnist.assembler import Assembler
from models_mnist.model import CorefNMN
from loader_mnist.data_reader import DataReader
from util import metrics
from util import support

# setting random seeds
np.random.seed(1234)
tf.set_random_seed(1234)

# Data files
args['data_root'] = os.path.join(args['data_root'], args['dataset'])
args['text_vocab_path'] = os.path.join(args['data_root'], 'vocabulary_mnist.txt')

root = args['data_root']
args['prog_vocab_path'] = os.path.join(root, 'vocabulary_layout_mnist.txt')
args['answer_list_path'] = os.path.join(root, 'answers_mnist.txt')
imdb_path_train = os.path.join(root, 'imdb_train.npy')

# assemblers for question and caption programs
question_assembler = Assembler(args['prog_vocab_path'])
assemblers = {'ques': question_assembler}

# Dataloader for train
input_dict = {'path': imdb_path_train, 'shuffle': True, 'one_pass': False,
              'assembler': question_assembler, 'use_count': False,
              'args': args}
train_loader = DataReader(input_dict)

# model params for training
train_params = args.copy()
# use the ground truth program for training
train_params['use_gt_prog'] = True
train_params['text_vocab_size'] = train_loader.batch_loader.vocab_dict.num_vocab
train_params['prog_vocab_size'] = len(question_assembler.module_names)
train_params['pad_id'] = train_loader.batch_loader.vocab_dict.word2idx('<pad>')
train_params['num_rounds'] = train_loader.batch_loader.num_rounds
train_params['num_choices'] = train_loader.num_choices
print('Using a vocab size: %d' % train_params['text_vocab_size'])

# model for training
model = CorefNMN(train_params, assemblers)
model.setup_training()

# train with Adam, optimization ops
solver = tf.train.AdamOptimizer(learning_rate=train_params['learning_rate'])
gradients = solver.compute_gradients(model.get_total_loss())

# clip gradients based on value
gradients = [(tf.clip_by_value(g, -2.0, 2.0), v) if g is not None else (g, v)
             for g, v in gradients]
solver_op = solver.apply_gradients(gradients)
# add it to the output
model.add_solver_op(solver_op)

# adjust snapshot to have a time stamp folder
cur_time = time.strftime('%a-%d%b%y-%X', time.gmtime())
args['snapshot_path'] = os.path.join(args['snapshot_path'], cur_time)
os.makedirs(args['snapshot_path'], exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
print('Saving checkpoints at: %s' % args['snapshot_path'])

# initialize all variables
sess.run(tf.global_variables_initializer())

# forget about embed and module scopes
del train_params['embed_scope']
if 'module_scope' in train_params:
  del train_params['module_scope']
#-------------------------------------------------------------------------

print('Running training iteration..')
num_iter_per_epoch = int(train_loader.batch_loader.num_inst/args['batch_size'])
print('Number of iterations per epoch: %d' % num_iter_per_epoch)

# exponential smoothing for loss
smoother = metrics.ExponentialSmoothing()

for n_iter, batch in enumerate(train_loader.batches()):
  # add epoch and iteration
  epoch = float(n_iter) / num_iter_per_epoch
  batch['epoch'] = epoch
  batch['n_iter'] = n_iter

  if n_iter >= args['num_epochs'] * num_iter_per_epoch:
    break

  # perform training iteration
  losses, _ = model.run_train_iteration(batch, sess)
  losses = smoother.report(losses)

  # printing log
  if n_iter % 10 == 0:
    cur_time = time.strftime('%a %d%b%y %X', time.gmtime())
    print_format = ('[%s][It: %d][Ep: %.2f][Loss: %.3f Prog: %.3f Ans: %.3f]')
    print_info = (cur_time, n_iter, epoch, losses['total'], losses['prog'],
                  losses['ans'])
    print(print_format % print_info)

  # save snapshot after every epoch
  if n_iter % num_iter_per_epoch == 0:
    epoch = float(n_iter) / num_iter_per_epoch

    # Save snapshot at every epoch
    file_name = 'model_epoch_%03d.tmodel' % epoch
    snapshot_path = os.path.join(args['snapshot_path'], file_name)
    snapshot_saver.save(sess, snapshot_path, write_meta_graph=False)

    # also save the arguments
    params_path = snapshot_path.replace('.tmodel', '_params.json')
    with open(params_path, 'w') as file_id:
      json.dump(train_params, file_id)
    print('Snapshot saved to: ' + snapshot_path)
