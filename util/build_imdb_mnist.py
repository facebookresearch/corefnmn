"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Final preprocessing script to create the image dialog database that
can be used to serve batches by the batch loader while training and evaluation
for MNIST experiments.
"""

import argparse
from collections import defaultdict
import copy
import json
import os
import pdb
import sys

import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm as progressbar

from util import text_processing, clean, support


# program supervision
# question types vs layouts (manually done)
prog_ques_type = {
                  'Qa': '_Find _Exist',
                  'Qb': '_Find _Count',
                  'Qc': '_Find _Describe',
                  'Qd': '_Refer _Transform _Describe',
                  'Qe': '_Refer _Not _Find _And _Exist'
                 }


def build_imdb(data, split, vocab, ans_list, FLAGS):
  """Function to build the image dialog dataset, given the data split.

  Args:
    data: MNIST Dialog dataset json
    split: Data split -- train | valid | test
    vocab: Vocabulary object created from question vocabulary (train only)
    ans_list: List of answers, created from train set
    FLAGS: Command line arguments

  Returns:
    imdb: Image dialog database to train corefnmn
  """

  print('Building imdb for %s' % split)
  source = data['%sExamples' % split]
  ans_dict = {word: ii for ii, word in enumerate(ans_list)}

  # process and tokenize all questions and answers
  tokenizer = lambda x: [vocab.word2idx(ii) for ii in
                         word_tokenize(clean.clean_non_ascii(x))]

  print('Collecting and tokenizing questions')
  ques_dict = {}
  ques_list = []

  for datum in progressbar(source):
    for round_datum in datum['qa']:
      ques = round_datum['question']
      if ques in ques_dict: continue
      else:
        ques_list.append(ques)
        ques_dict[ques] = len(ques_dict)

  clean_ques = [tokenizer(ques.lower()) for ques in progressbar(ques_list)]
  max_ques_len = max([len(ii) for ii in clean_ques])

  ques_tokens = np.zeros((len(clean_ques), max_ques_len)).astype('int32')
  ques_tokens.fill(vocab.word2idx('<pad>'))
  ques_lens = np.zeros(len(clean_ques)).astype('int32')

  for q_id, tokens in progressbar(enumerate(clean_ques)):
    ques_lens[q_id] = len(tokens)
    ques_tokens[q_id, :ques_lens[q_id]] = np.array(tokens)
  #--------------------------------------------------------------------------

  imdb = {}
  # number of entries in the database
  num_dialogs = len(source)
  imdb['data'] = [None] * num_dialogs
  imdb['ans_inds'] = ans_list
  imdb['ques'], imdb['ques_len'] = ques_tokens, ques_lens
  #--------------------------------------------------------------------------

  for dialog_id, datum in progressbar(enumerate(source)):
    img_id = datum['img']
    img_path = os.path.join(FLAGS.image_root, split, '%05d.jpg' % img_id)

    # compact bundle with all the information
    bundle = {'image_name': img_id, 'image_path': img_path,
              'question_id': [], 'question_ind': [], 'answer_ind': [],
              'gt_layout_tokens': []}

    # bundle as questions in a conversation together
    for r_id, round_data in enumerate(datum['qa']):
      q_id = img_id * 10 + r_id

      bundle['question_id'].append(q_id)
      ques_ind = ques_dict[round_data['question']]
      bundle['question_ind'].append(ques_ind)
      answer = ans_dict.get(round_data['answer'], '<unk>')
      bundle['answer_ind'].append(answer)

      # sanity check
      if answer == '<unk>':
        print(answer)

      # layout
      layout = prog_ques_type[round_data['metaInfo'][0]]

      # replace find with refer
      if r_id > 0 and round_data['metaInfo'][0] in ['Qa', 'Qb']:
        layout = layout.replace('_Find', '_Refer _Find _And');
      if r_id > 0 and round_data['metaInfo'][0] == 'Qc':
        layout = layout.replace('_Find', '_Refer');

      """Layout modifications for NMN version (baseline)
      if round_data['metaInfo'][0] == 'Qd':
        layout = layout.replace('Refer', 'Find')
      if round_data['metaInfo'][0] == 'Qe':
        layout = '_Find _Exist'
      """
      # layout for independent questions
      bundle['gt_layout_tokens'].append(layout)

    # record
    imdb['data'][dialog_id] = bundle

  return imdb


def save_vocabularies(train_examples, FLAGS):
  """Extract and save vocabularies for questions and answers.

  Args:
    train_examples: Training examples

  Returns:
    words: Vocabulary (dictionary) extracted from the questions
    ans_list: List of possible answers, extracted from train set
  """

  words = {}
  ans_list = {}
  for datum in progressbar(train_examples):
    for ques_datum in datum['qa']:
      token = ques_datum['answer'].lower()
      words[token] = words.get(token, 0) + 1
      ans_list[token] = 1

      for token in word_tokenize(ques_datum['question']):
        token = token.lower()
        words[token] = words.get(token, 0) + 1

  # additional tokens
  words['<pad>'] = 1
  words['<start>'] = 1
  words['<end>'] = 1
  words['<unk>'] = 1

  print('Saving to: ' + FLAGS.vocab_save_path)
  with open(FLAGS.vocab_save_path, 'w') as file_id:
    file_id.write('\n'.join(sorted(words.keys())))

  # answer lists
  ans_list = list(ans_list.keys())
  ans_list.append('<unk>')
  print('Saving to: ' + FLAGS.answers_save_path)
  with open(FLAGS.answers_save_path, 'w') as file_id:
    file_id.write('\n'.join(ans_list))


def save_mean_std_image(FLAGS):
  """Compute and save mean and std image from train images.

  Args:
    FLAGS: Commandline arguments
  """

  import pdb
  image_list = os.listdir(os.path.join(FLAGS.image_root, 'train'))

  # compute the mean of the train images and save
  mean_img = None
  std_img = None
  for image_name in progressbar(image_list):
    image_path = os.path.join(FLAGS.image_root, 'train', image_name)
    image = support.load_image(image_path)

    if mean_img is None:
      mean_img = image
      std_img = image ** 2
    else:
      mean_img += image
      std_img += image ** 2

  mean_img = mean_img / len(image_list)
  std_img = std_img / len(image_list)

  mean_img = np.mean(np.mean(mean_img, 0), 0)
  std_img = np.mean(np.mean(std_img, 0), 0)
  std_img = np.sqrt(std_img - mean_img ** 2)

  print('Saving mean and std at: %s' % FLAGS.mean_save_path)
  np.save(FLAGS.mean_save_path, {'mean_img': mean_img, 'std_img': std_img})


def main(FLAGS):
  """Main function.

  1. Extracts vocabularies from questions and answers.
  2. Creates and saves image dialog databases for train | valid | test splits.

  Args:
    FLAGS: Command-line options.
  """

  # Read the dataset.
  with open(FLAGS.json_path) as file_id:
    data = json.load(file_id)

  # Extract vocabulary and answer list.
  save_vocabularies(data['trainExamples'], FLAGS)

  # Extract mean and std of train images.
  save_mean_std_image(FLAGS)

  # Read the vocabulary files (questions | answers) and create objects
  vocab = text_processing.VocabDict(FLAGS.vocab_save_path)

  with open(FLAGS.answers_save_path, 'r') as file_id:
    ans_list = [ii.strip('\n') for ii in file_id.readlines()]

  # data splits
  for split in ['train', 'valid', 'test']:
    imdb_split = build_imdb(data, split, vocab, ans_list, FLAGS)
    save_path = os.path.join(FLAGS.imdb_save_path, 'imdb_%s.npy' % split)
    print('Saving imdb build: %s' % save_path)
    np.save(save_path, np.array(imdb_split))


if __name__ == '__main__':
  title = 'Process all the information into a database for easier access'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--json_path', required=True,
                      help='Path to MNIST Dialog dataset json file')
  parser.add_argument('--image_root', required=True,
                      help='Path to root folder of all the images')
  parser.add_argument('--vocab_save_path', required=True,
                      help='Path to save the vocabulary from training set')
  parser.add_argument('--answers_save_path', required=True,
                      help='Path to save the answers file from training set')
  parser.add_argument('--imdb_save_path', required=True,
                      help='Path to save the image dialog dataset')
  parser.add_argument('--mean_save_path', required=True,
                      help='Path to save the mean and std of train images')
  FLAGS = parser.parse_args()

  main(FLAGS)
