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
can be used to serve batches by the batch loader while training and evaluation.
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

from util import text_processing, clean


stop_words = ['the', 'a', 'an', 'you', 'was', 'and', 'are']
def build_imdb(FLAGS):
  """Method to construct and save the image-database for the dataset
  """

  print('Building imdb for visdial split: %s' % FLAGS.visdial_file)
  qid2layout_dict = np.load(FLAGS.ques_prog_file)[()]

  ques_att_file = FLAGS.ques_prog_file.replace('.layout', '.attention')
  ques_prog_att = np.load(ques_att_file)[()]

  cap_progs = np.load(FLAGS.cap_prog_file)[()]
  cap_att_file = FLAGS.cap_prog_file.replace('.layout', '.attention')
  cap_prog_att = np.load(cap_att_file)[()]
  vocab = text_processing.VocabDict(FLAGS.vocab_file)

  # load the data
  with open(FLAGS.visdial_file, 'r') as file_id:
    vd_data = json.load(file_id)

  # load the reference data
  with open(FLAGS.coreference_file, 'r') as file_id:
    references = json.load(file_id)
    references = references['data']['dialogs']

  # coco_name = img_split + '2014'
  # img_root = os.path.abspath(image_dir % coco_name)
  # feat_root = os.path.abspath(feature_dir % coco_name)
  # img_name_format = 'COCO_' + coco_name + '_%012d'

  # process and tokenize all questions and answers
  tokenizer = lambda x, suff: [vocab.word2idx(ii) for ii in
                               word_tokenize(clean.clean_non_ascii(x + suff))]

  print('Tokenizing captions')
  caption_list = [ii['caption'] for ii in vd_data['data']['dialogs']]
  clean_cap = [tokenizer(cap, '') for cap in progressbar(caption_list)]
  max_cap_len = max([len(ii) for ii in clean_cap])

  cap_tokens = np.zeros((len(clean_cap), max_cap_len)).astype('int32')
  cap_tokens.fill(vocab.word2idx('<pad>'))
  cap_lens = np.zeros(len(clean_cap)).astype('int32')

  for q_id, tokens in progressbar(enumerate(clean_cap)):
    cap_lens[q_id] = len(tokens)
    cap_tokens[q_id, :cap_lens[q_id]] = np.array(tokens)

  print('Tokenizing questions')
  question_list = vd_data['data']['questions']
  clean_ques = [tokenizer(ques, '?') for ques in progressbar(question_list)]
  max_ques_len = max([len(ii) for ii in clean_ques])

  ques_tokens = np.zeros((len(clean_ques), max_ques_len)).astype('int32')
  ques_tokens.fill(vocab.word2idx('<pad>'))
  ques_lens = np.zeros(len(clean_ques)).astype('int32')

  for q_id, tokens in progressbar(enumerate(clean_ques)):
    ques_lens[q_id] = len(tokens)
    ques_tokens[q_id, :ques_lens[q_id]] = np.array(tokens)

  print('Tokenizing answers')
  answer_list = vd_data['data']['answers']
  clean_ans = [tokenizer(ans, '') for ans in progressbar(answer_list)]
  max_ans_len = max([len(ii) for ii in clean_ans])

  ans_tokens = np.zeros((len(clean_ans), max_ans_len)).astype('int32')
  ans_tokens.fill(vocab.word2idx('<pad>'))
  ans_lens = np.zeros(len(clean_ans)).astype('int32')

  ans_in = np.zeros((len(clean_ans), max_ans_len + 1)).astype('int32')
  ans_out = np.zeros((len(clean_ans), max_ans_len + 1)).astype('int32')
  ans_in.fill(vocab.word2idx('<pad>'))
  ans_out.fill(vocab.word2idx('<pad>'))
  start_token_id = vocab.word2idx('<start>')
  end_token_id = vocab.word2idx('<end>')
  ans_in[:, 0] = start_token_id

  for a_id, tokens in progressbar(enumerate(clean_ans)):
    ans_lens[a_id] = len(tokens)
    answer = np.array(tokens)
    ans_tokens[a_id, :ans_lens[a_id]] = answer
    ans_in[a_id, 1:ans_lens[a_id]+1] = answer
    ans_out[a_id, :ans_lens[a_id]] = answer
    ans_out[a_id, ans_lens[a_id]] = end_token_id

  ans_lens += 1

  imdb = {}
  # number of entries in the database
  num_dialogs = len(vd_data['data']['dialogs'])
  imdb['data'] = [None] * num_dialogs
  imdb['ans'], imdb['ans_len'] = ans_tokens, ans_lens
  imdb['ans_in'], imdb['ans_out'] = ans_in, ans_out
  imdb['ques'], imdb['ques_len'] = ques_tokens, ques_lens
  imdb['cap'], imdb['cap_len'] = cap_tokens, cap_lens
  imdb['cap_prog'], imdb['cap_prog_att'] = cap_progs, np.array(cap_prog_att)

  for dialog_id, datum in progressbar(enumerate(vd_data['data']['dialogs'])):
    img_id = datum['image_id']
    img_path = FLAGS.image_path_format % img_id
    feat_path = FLAGS.feature_path % img_id

    # compact bundle with all the information
    bundle = {'image_name': img_id, 'image_path': img_path,
              'feature_path': feat_path, 'caption_ind': dialog_id,
              'question_id': [], 'question_ind': [], 'answer_ind': [],
              'option_ind': [], 'gt_ind' : [], 'gt_layout_tokens': [],
              'gt_layout_att': []}

    # reference datum
    refer_datum = references[dialog_id]
    assert(refer_datum['image_id'] == img_id)
    # for each cluster, get the first mention
    clusters = {}
    caption_clusters = (refer_datum['caption_reference_clusters'] +
                        refer_datum['caption_coref_clusters'])
    for ii in caption_clusters:
      c_id = ii['cluster_id']
      clusters[c_id] = clusters.get(c_id, 'c')

    # each round
    for r_id in range(10): # assuming 10 rounds for now
      referrer = refer_datum['dialog'][r_id]
      for ii in referrer['question_reference_clusters']:
        c_id = ii['cluster_id']
        clusters[c_id] = clusters.get(c_id, 'q%d' % r_id)

      for ii in referrer['answer_reference_clusters']:
        c_id = ii['cluster_id']
        # to distinguish answer
        clusters[c_id] = clusters.get(c_id, 'a%d' % r_id)

    # bundle as questions in a conversation together
    num_refers = 0
    for r_id, round_data in enumerate(datum['dialog']):
      q_id = img_id * 10 + r_id

      bundle['question_id'].append(q_id)
      bundle['question_ind'].append(round_data['question'])
      bundle['answer_ind'].append(round_data['answer'])
      bundle['option_ind'].append(round_data['answer_options'])
      bundle['gt_ind'].append(round_data['gt_index'])

      # gt attention for parsed layout
      attention = np.array(ques_prog_att[round_data['question']])

      # check if references is non-empty and replace with _Refer
      layout = copy.deepcopy(list(qid2layout_dict[q_id]))
      referrer = refer_datum['dialog'][r_id]['question_referrer_clusters']
      if len(referrer) > 0:
        refer = referrer[0]
        # pick _Find module with max attention overlap
        max_overlap = (0, 0)
        for pos, token in enumerate(layout):
          if token == '_Find':
            start = max(attention[pos][0], refer['start_word'])
            end = min(attention[pos][1], refer['end_word'])
            overlap = min(0, end - start)
            if max_overlap[1] < overlap: max_overlap = (pos, overlap)

        # reset it to _Refer
        pos, _ = max_overlap
        layout[pos] = '_Refer'
        attention[pos] = [refer['start_word'], refer['end_word']]

        # get that cluster id, and corresponding history attention
        num_refers += 1
      bundle['gt_layout_tokens'].append(layout)

      # check for the words attending to
      ques_tokens = imdb['ques'][round_data['question']]
      ques_words = [vocab.idx2word(ii) for ii in ques_tokens]
      for index, pos in enumerate(attention):
        # if single word,  'the', 'a', 'of', 'you'
        try:
          if (pos[1] - pos[0]) == 1 and ques_words[pos[0]] in stop_words:
            attention[index] = [0, 0]
        except: pdb.set_trace()
      bundle['gt_layout_att'].append(attention)

    # record
    imdb['data'][dialog_id] = bundle
  return imdb


if __name__ == '__main__':
  title = 'Process all the information into a database for easier access'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--ques_prog_file', required=True,
                      help='Path to question ground truth programs')
  parser.add_argument('--cap_prog_file', required=True,
                      help='Path to caption ground truth programs')
  parser.add_argument('--image_path_format', required=True,
                      help='Path to find the image given the COCO id')
  parser.add_argument('--feature_path', required=True,
                      help='Path to find the features given the COCO id')
  parser.add_argument('--coreference_file', required=True,
                      help='Visdial file infused with coreference supervision')
  parser.add_argument('--visdial_file', required=True,
                      help='Original visdial file')
  parser.add_argument('--vocab_file', required=True,
                      help='Visual Dialog vocabulary file')
  parser.add_argument('--save_path', required=True,
                      help='Path to save the image dialog dataset')
  FLAGS = parser.parse_args()

  imdb_data = build_imdb(FLAGS)
  print('Saving imdb build: %s' % FLAGS.save_path)
  np.save(FLAGS.save_path, np.array(imdb_data))
