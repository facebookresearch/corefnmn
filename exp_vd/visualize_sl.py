r"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script to visualize trained Visual Dialog model using supervised learning.

Visualizes visual dialog model that performs explicit visual coreference resolution
using neural module networks. Additional details are in the paper:
  Visual Coreference Resolution in Visual Dialog using Neural Module Networks
  Satwik Kottur, José M. F. Moura, Devi Parikh, Dhruv Batra, Marcus Rohrbach
  European Conference on Computer Vision (ECCV), 2018

Usage:
  python -u exp_vd/visualize_sl.py --gpu_id=0 --test_split='val' \
       --checkpoint='checkpoints/model_epoch_005.tmodel' --batch_size 1
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

from exp_vd import options

# Read command line options.
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help="Checkpoint to load")
parser.add_argument('--batch_size', type=int, default=10, 
                    help='Batch size for visualization')
parser.add_argument('--test_split', default='val',
                    help='Split to run visualization')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num_instances', type=int, default=50)

try:
  args = vars(parser.parse_args())
except (IOError) as msg:
  parser.error(str(msg))

# Set the cuda environment variable for the gpu to use.
gpu_id = '' if args['gpu_id'] < 0 else str(args['gpu_id'])
print('Using GPU id: %s' % gpu_id)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                           allow_soft_placement=False,
                           log_device_placement=False)
sess = tf.Session(config=tf_config)

from models_vd.assembler import Assembler
from models_vd.model import CorefNMN
from loader_vd.data_reader import DataReader
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
args['supervise_attention'] = False
print('Current model: ' + args['model'])
support.pretty_print_dict(args)

# Data files
root = args['data_root']
imdb_path_val = os.path.join(root, 'imdb_%s.npy' % args['test_split'])

# assemblers for question and caption programs
question_assembler = Assembler(args['prog_vocab_path'])
caption_assembler = Assembler(args['prog_vocab_path'])
assemblers = {'ques': question_assembler, 'cap': caption_assembler}

# dataloader for val
input_dict = {'path': imdb_path_val, 'shuffle': False, 'one_pass': True,
              'args': args, 'assembler': question_assembler,
              'fetch_options': True}
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
ranks = []
matches = []
cur_iter = 0
to_save = {'output': [], 'batch': []}

for batch in progressbar(val_loader.batches(), total=args['num_instances']):
  _, outputs = model.run_visualize_iteration(batch, sess)

  to_save['output'].append(outputs)
  to_save['batch'].append(batch)

  cur_iter += 1
  if cur_iter >= args['num_instances']:
    break

# Save the output + batch
batch_path = '{0}.{1}_batches.npy'.format(args['checkpoint'],
                                          args['num_instances'])
support.save_batch(to_save, batch_path)
