"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script to train Visual Dialog model using supervised learning.

Trains visual dialog model that performs explicit visual coreference resolution
using neural module networks. Additional details are in the paper:
 Visual Coreference Resolution in Visual Dialog using Neural Module Networks
 Satwik Kottur, José M. F. Moura, Devi Parikh, Dhruv Batra, Marcus Rohrbach
 European Conference on Computer Vision (ECCV), 2018

Usage:
 python -u exp_mnist/eval_sl.py --gpu_id=0 --test_split='valid' \
   --checkpoint='checkpoints/model_epoch_005.tmodel'
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
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--test_split', default='valid', \
                    help='Which split to run evaluation on')
parser.add_argument('--gpu_id', type=int, default=0)

try:
  args = vars(parser.parse_args())
except (IOError) as msg:
  parser.error(str(msg))

# set the cuda environment variable for the gpu to use
gpu_id = '' if args['gpu_id'] < 0 else str(args['gpu_id'])
print('Using GPU id: %s' % gpu_id)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

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

# read the train args from checkpoint
param_path = args['checkpoint'].replace('.tmodel', '_params.json')
with open(param_path, 'r') as file_id:
  saved_args = json.load(file_id)

saved_args.update(args)
args = saved_args
args['preload_feats'] = False
# no supervision is needed
args['supervise_attention'] = False

support.pretty_print_dict(args)

# Data files
root = args['data_root']
imdb_path_val = os.path.join(root, 'imdb/imdb_%s.npy' % args['test_split'])

# assembler
question_assembler = Assembler(args['prog_vocab_path'])
assemblers = {'ques': question_assembler}

# dataloader for val
input_dict = {'path': imdb_path_val, 'shuffle': False, 'one_pass': True,
              'args': args, 'assembler': question_assembler,
              'use_count': False, 'fetch_options': True}
val_loader = DataReader(input_dict)

# model for training
eval_params = args.copy()
eval_params['use_gt_prog'] = False # for training
eval_params['enc_dropout'] = False
eval_params['dec_dropout'] = False
eval_params['dec_sampling'] = False # do not sample, take argmax

# for models trained later
if 'num_rounds' not in eval_params:
  eval_params['num_rounds'] = val_loader.batch_loader.num_rounds

# model for evaluation
# create another assembler of caption
model = CorefNMN(eval_params, assemblers)

# Load snapshot
print('Loading checkpoint from: %s' % args['checkpoint'])
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
snapshot_saver.restore(sess, args['checkpoint'])

print('Evaluating on %s' % args['test_split'])
ans_matches = []
prog_matches = []
total_iter = int(val_loader.batch_loader.num_inst / args['batch_size'])
num_iters = 0
for batch in progressbar(val_loader.batches(), total=total_iter):
  batch_matches, outputs = model.run_evaluate_iteration(batch, sess)

  ans_matches.append(batch_matches)
  if 'matches' in outputs:
    prog_matches.append(outputs['matches'])

try:
  if len(prog_matches) > 0:
    prog_matches = np.concatenate(prog_matches)
    percent = 100*np.sum(prog_matches) / prog_matches.size
    print('Program accuracy: %f percent\n' % percent)
except:
  pass

ans_matches = np.concatenate(ans_matches)
percent = 100 * np.sum(ans_matches) / ans_matches.size
print('Answer accuracy: %f percent\n' % percent)
