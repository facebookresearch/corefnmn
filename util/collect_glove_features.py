"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Given the data file, create a vocabulary file and extract the glove features
for embedding initializations.
"""

import argparse
from collections import defaultdict
import json
import re
import sys
from unidecode import unidecode

import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import numpy as np
import spacy


def main(args):
  # initialize vocab from file
  print('Reading vocabulary from: %s' % args.vocab_file)
  with open(args.vocab_file, 'r') as fileId:
    vocab_dict = json.load(fileId)
  vocab_set = set(vocab_dict['word2ind'].keys())

  # Though we have collected all the words from source vocabulary, add <UNK>
  # and add other tokens for answer decoding
  # <start> <end> <pad>
  vocab_set.add('<unk>')
  vocab_set.add('<start>')
  vocab_set.add('<end>')
  vocab_set.add('<pad>')
  print('Vocabulary size: %d, keeping all of them ..' % len(vocab_set))
  vocab_list = list(vocab_set)
  vocab_list.sort()

  print('Saving vocabulary: ' + args.save_path)
  with open(args.save_path, 'w') as file_id:
    file_id.writelines([w.replace('\u2019', '') + '\n' for w in vocab_list])

  # Collect glove vectors for the words, and save.
  glove_dim = 300
  glove_mat = np.zeros((len(vocab_list), glove_dim), np.float32)
  nlp = spacy.load('en_vectors_web_lg')
  for index, word in enumerate(vocab_list):
    glove_mat[index] = nlp(word).vector

  glove_mat_file = args.save_path.replace('.txt', '_glove.npy')
  print('Saving glove vectors: ' + glove_mat_file)
  np.save(glove_mat_file, glove_mat)


if __name__ == '__main__':
  title = 'Restructure Stanford Parser to a single line'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--vocab_file', required=True,
                      help='Vocabulary file from original visdial code')
  parser.add_argument('--save_path', required=True,
                      help=('Path to save the vocabulary text file and '
                            'glove embeddings for corefnmn code'))
  args = parser.parse_args()

  main(args)
