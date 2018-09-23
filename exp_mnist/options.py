"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script to read command line flags.

Uses argparse library to read command line flags.
Author: Satwik Kottur
"""

import argparse
import os
import pdb
from util import support

# read command line arguments
def read_command_line():
  title = 'Train explicit coreference resolution visual dialog model'
  parser = argparse.ArgumentParser(description=title)
  #-------------------------------------------------------------------------

  # data input settings
  parser.add_argument('--dataset', default='mnist', help='Visdial dataset type')
  parser.add_argument('--input_img', default='data/resnet_res5c/',\
                      help='Path with image features')
  parser.add_argument('--data_root', default='data/',\
                      help='HDF5 file with preprocessed questions')
  parser.add_argument('--text_vocab_path', default='',
                      help='Path to the vocabulary for text')
  parser.add_argument('--prog_vocab_path', default='',
                      help='Path to the vocabulary for programs')
  parser.add_argument('--snapshot_path', default='checkpoints/',
                      help='Path to save checkpoints')
  #--------------------------------------------------------------------------

  # specify encoder/decoder
  parser.add_argument('--model', default='nmn', help='Name of the model')
  parser.add_argument('--generator', default='ques',
                      help='Name of the generator to use (ques | memory)')
  parser.add_argument('--img_norm', default=1, type=int,
                      help='Normalize the image feature. 1=yes, 0=no')
  #-------------------------------------------------------------------------

  # model hyperparameters
  parser.add_argument('--h_feat', default=7, type=int,
                      help='Height of visual conv feature')
  parser.add_argument('--w_feat', default=7, type=int,
                      help='Width of visual conv feature')
  parser.add_argument('--d_feat', default=64, type=int,
                      help='Size of visual conv feature')
  parser.add_argument('--text_embed_size', default=32, type=int,
                      help='Size of embedding for text')
  parser.add_argument('--map_size', default=128, type=int,
                      help='Size of the final mapping')
  parser.add_argument('--prog_embed_size', default=32, type=int,
                      help='Size of embedding for program tokens')
  parser.add_argument('--lstm_size', default=64, type=int,
                      help='Size of hidden state in LSTM')
  parser.add_argument('--enc_dropout', default=True, type=bool,
                      help='Dropout in encoder')
  parser.add_argument('--dec_dropout', default=True, type=bool,
                      help='Dropout in decoder')
  parser.add_argument('--num_layers', default=1, type=int,
                      help='Number of layers in LSTM')

  parser.add_argument('--max_enc_len', default=14, type=int,
                      help='Maximum encoding length for sentences (ques|cap)')
  parser.add_argument('--max_dec_len', default=8, type=int,
                      help='Maximum decoding length for programs (ques|cap)')
  parser.add_argument('--dec_sampling', default=False, type=bool,
                      help='Sample while decoding program')
  parser.add_argument('--use_batch_norm', dest='use_batch_norm',
                      action='store_true', help='Flag to use batch norm')
  parser.set_defaults(use_batch_norm=False)
  parser.add_argument('--align_image_features', dest='align_image_features',
                      action='store_true', help='Flag to align image features')
  parser.set_defaults(align_image_features=False)
  parser.add_argument('--use_refer', dest='use_refer',
                      action='store_true', help='Flag for Refer Module')
  parser.set_defaults(use_refer=False)
  parser.add_argument('--remove_aux_find', dest='remove_aux_find',
                      action='store_true',
                      help='Flag to remove auxilliary find modules')
  parser.set_defaults(remove_aux_find=False)
  parser.add_argument('--use_fact', dest='use_fact',
                      action='store_true', help='Flag to use Q+A as fact')
  parser.set_defaults(use_fact=False)
  parser.add_argument('--amalgam_text_feats', dest='amalgam_text_feats',
                      action='store_true',
                      help='Flag to amalgamate text features')
  parser.set_defaults(amalgam_text_feats=False)
  #-------------------------------------------------------------------------

  # optimization params
  parser.add_argument('--batch_size', default=20, type=int,
                      help='Training batch size (adjust based on GPU memory)')
  parser.add_argument('--learning_rate', default=1e-4, type=float,
                      help='Learning rate for training')
  parser.add_argument('--dropout', default=0.5, type=float, help='Dropout')
  parser.add_argument('--num_epochs', default=20, type=int,
                      help='Maximum number of epochs to run training')
  parser.add_argument('--gpu_id', type=int, default=0,
                      help='GPU id to use for training, -1 for CPU')
  #-------------------------------------------------------------------------

  try:
    parsed_args = vars(parser.parse_args())
  except (IOError) as msg:
    parser.error(str(msg))

  # set the cuda environment variable for the gpu to use
  gpu_id = '' if parsed_args['gpu_id'] < 0 else str(parsed_args['gpu_id'])
  print('Using GPU id: %s' % gpu_id)
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

  # pretty print arguments and return
  support.pretty_print_dict(parsed_args)

  return parsed_args
