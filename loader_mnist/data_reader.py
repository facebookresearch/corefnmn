"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Dataloader file for Visual Dialog experiments.

Explicit visual coreference resolution in visual dialog using neural module
networks.

Author: Satwik Kottur
"""

from __future__ import absolute_import, division, print_function

import h5py
import json
import os
import threading
import queue
import numpy as np
from tqdm import tqdm as progressbar

from util import text_processing, support


class BatchLoaderMNIST:
  """Subclass to DataReader that serves batches during training.
  """
  def __init__(self, imdb, params):
    """Initialize by reading the data and pre-processing it.
    """

    self.imdb = imdb
    self.params = params
    self.fetch_options = self.params.get('fetch_options', False)
    self.num_inst = len(self.imdb['data'])
    self.num_rounds = len(self.imdb['data'][0]['question_ind'])

    # load vocabulary
    vocab_path = params['text_vocab_path']
    self.vocab_dict = text_processing.VocabDict(vocab_path)
    self.T_encoder = params['max_enc_len']

    # record special token ids
    self.start_token_id = self.vocab_dict.word2idx('<start>')
    self.end_token_id = self.vocab_dict.word2idx('<end>')
    self.pad_token_id = self.vocab_dict.word2idx('<pad>')
    # Load answers
    with open(params['args']['answer_list_path'], 'r') as file_id:
      choices = [ii.strip('\n') for ii in file_id.readlines()]
      self.num_choices = len(choices)
      self.choices2ind = {ii: index for index, ii in enumerate(choices)}
      self.ind2choices = {index: ii for index, ii in enumerate(choices)}

    # peek one example to see whether answer and gt_layout are in the data
    test_data = self.imdb['data'][0]
    self.load_gt_layout = test_data.get('gt_layout_tokens', False)
    if 'load_gt_layout' in params:
      self.load_gt_layout = params['load_gt_layout']

    self.T_decoder = params['max_dec_len']
    self.assembler = params['assembler']

    # load the mean of the images
    load_path = params['path'].split('/')[:-1] + ['train_image_mean.npy']
    load_path = '/'.join(load_path)
    print('Loading training image stats from: ' + load_path)
    img_stats = np.load(load_path)[()]
    mean_img = img_stats['mean_img'].reshape([1, 1, -1])
    std_img = img_stats['std_img'].reshape([1, 1, -1])

    # read all the images
    images = {}
    print('Reading images..')
    for datum in progressbar(self.imdb['data']):
      img_path = datum['image_path']
      if img_path not in images:
        cur_img = support.load_image(img_path) / 255.
        cur_img = (cur_img - mean_img) / std_img
        images[img_path] = cur_img

    self.images = images

    # get the shape from random image
    for _, sample in self.images.items():
      self.img_size = sample.shape
      break

    # convert to tokens
    self.digitizer = lambda x: [self.vocab_dict.word2idx(w) for w in x]

    # use history if needed by the program generator
    self.use_history = self.params['generator'] == 'mem'
    if self.use_history:
      self._construct_history()

    # if fact is to be used
    if self.params['use_fact']:
      self._construct_fact()
  #--------------------------------------------------------------------------

  def _construct_fact(self):
    """Method to construct facts.

    Facts are previous question and answers strings concatenated as one. These
    serve as memory units that the model can refer back to.

    For example, 'Q: What is the man wearing? A: Sweater.' will have a fact
    'What is the man wearing? Sweater.' so that the model can address follow-up
    questions like 'What color is it?' by referring to this fact.
    """

    print('Constructing facts..')

    num_diags = len(self.imdb['data'])
    max_len = self.T_encoder + 1 # question + answer appended
    num_rounds = len(self.imdb['data'][0]['question_ind'])
    fact = np.zeros((num_diags, num_rounds, max_len))
    fact_len = np.zeros((num_diags, num_rounds))
    fact.fill(self.pad_token_id)

    for diag_id, datum in enumerate(self.imdb['data']):
      for r_id in range(num_rounds - 1):
        q_id = datum['question_ind'][r_id]
        a_id = datum['answer_ind'][r_id]
        ques, q_len = self.imdb['ques'][q_id], self.imdb['ques_len'][q_id]
        ans = self.vocab_dict.word2idx(self.ind2choices[a_id])

        # handle overflow
        bound = min(q_len, max_len)
        fact[diag_id, r_id, :bound] = ques[:bound]
        if bound < max_len:
          fact[diag_id, r_id, bound] = ans
        fact_len[diag_id, r_id] = bound + 1

    # flatten
    self.imdb['fact'] = fact
    self.imdb['fact_len'] = fact_len
  #--------------------------------------------------------------------------

  def _construct_history(self):
    """Method to construct history, which concatenates entire dialogs so far.
    """

    print('Constructing history..')

    num_diags = len(self.imdb['data'])
    max_len = self.T_encoder + 1 # question + answer appended
    num_rounds = len(self.imdb['data'][0]['question_ind'])
    history = np.zeros((num_diags, num_rounds, max_len))
    hist_len = np.zeros((num_diags, num_rounds))
    history.fill(self.pad_token_id)

    for diag_id, datum in enumerate(self.imdb['data']):
      for r_id in range(num_rounds - 1):
        q_id = datum['question_ind'][r_id]
        a_id = datum['answer_ind'][r_id]
        ques, q_len = self.imdb['ques'][q_id], self.imdb['ques_len'][q_id]
        ans = self.vocab_dict.word2idx(self.ind2choices[a_id])

        # handle overflow
        bound = min(q_len, max_len)
        history[diag_id, r_id, :bound] = ques[:bound]
        if bound < max_len:
          history[diag_id, r_id, bound] = ans
        hist_len[diag_id, r_id] = bound + 1

    self.imdb['hist'] = history
    self.imdb['hist_len'] = hist_len
  #--------------------------------------------------------------------------

  def load_one_batch(self, sample_ids):
    """Load data given the sample ids.
    """

    actual_batch_size = len(sample_ids)
    batch = {}

    # replace question _Find with _Refer
    find_module_token = self.assembler.name2idx_dict['_Find']
    #refer_module_token = self.assembler.name2idx_dict['_Refer']
    eos_token = self.assembler.name2idx_dict['<eos>']
    num_rounds = self.num_rounds

    # questions
    ques_inds = [jj for ii in sample_ids
                 for jj in self.imdb['data'][ii]['question_ind']]
    ques_batch = self.imdb['ques'][ques_inds][:, :self.T_encoder].transpose()
    ques_len = self.imdb['ques_len'][ques_inds]
    ques_ids = [jj for ii in sample_ids
                for jj in self.imdb['data'][ii]['question_id']]

    # answers
    ans_inds_batch = [jj for ii in sample_ids
                      for jj in self.imdb['data'][ii]['answer_ind']]

    image_path = [None] * actual_batch_size

    # load fact
    if self.params['use_fact']:
      fact = self.imdb['fact'][sample_ids]
      fact_len = self.imdb['fact_len'][sample_ids]
      # flatten
      fact = np.reshape(fact, [-1, fact.shape[-1]])
      fact_len = np.reshape(fact_len, [-1])
    else:
      fact, fact_len = None, None

    # programs
    if self.load_gt_layout:
      gt_layout_batch = np.zeros((self.T_decoder,
                                  num_rounds * actual_batch_size), np.int32)
      gt_layout_batch.fill(eos_token)

    # if features are needed, load images
    if 'prog' in self.params['model']:
      image_feats = np.zeros((actual_batch_size,) + self.img_size, np.float32)

    for n in range(len(sample_ids)):
      iminfo = self.imdb['data'][sample_ids[n]]


      image_path[n] = iminfo['image_path']
      image_feats[n] = self.images[iminfo['image_path']]

      # programs
      if self.load_gt_layout:
        # go over all the questions
        for r_id, layout in enumerate(iminfo['gt_layout_tokens']):
          split_layout = layout.split(' ')
          gt_layout_batch[:, num_rounds * n + r_id] = \
                self.assembler.module_list2tokens(split_layout,
                                                  self.T_decoder)

    # if history is needed
    if self.use_history:
      history = self.imdb['hist'][sample_ids]
      hist_len = self.imdb['hist_len'][sample_ids]
    else:
      history, hist_len = None, None

    batch = {'ques': ques_batch, 'ques_len': ques_len,
             'fact': fact, 'fact_len': fact_len,
             'hist': history, 'hist_len': hist_len,
             'ans_ind': ans_inds_batch,
             'img_path': image_path, 'imgs': image_feats,
             'ques_id': ques_ids, 'gt_layout': gt_layout_batch}

    return batch

class DataReader:
  """Main dataloader class for experiments on Visual Dialog.
  """

  def __init__(self, params):
    imdb_path = params['path']
    print('Loading imdb from: %s' % params['path'], end='')
    if imdb_path.endswith('.npy'): imdb = np.load(imdb_path)
    else: raise TypeError('unknown imdb format.')
    self.imdb = imdb[()]

    self.shuffle = params.get('shuffle', True)
    self.one_pass = params.get('one_pass', False)
    self.prefetch_num = params.get('num_prefetch', 8)
    self.params = params
    copy_args = {'max_enc_len', 'max_dec_len', 'text_vocab_path', 'model',
                 'fix_ques_layout', 'fix_cap_layout', 'batch_size', 'use_fact',
                 'supervise_attention', 'answer_list_path', 'generator'}
    self.params.update({ii:params['args'][ii] for ii in copy_args
                        if ii in params['args'] and
                      params['args'][ii] is not None})

    # MNIST data loader
    self.batch_loader = BatchLoaderMNIST(self.imdb, self.params)
    self.num_choices = self.batch_loader.num_choices

    # Start prefetching thread
    self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
    self.prefetch_thread = threading.Thread(target=_run_prefetch,
      args=(self.prefetch_queue, self.batch_loader, self.imdb,
         self.shuffle, self.one_pass, self.params))
    self.prefetch_thread.daemon = True
    self.prefetch_thread.start()

  def batches(self):
    while True:
      # Get a batch from the prefetching queue
      if self.prefetch_queue.empty(): pass
        #print('data reader: waiting for data loading (IO is slow)...')
      batch = self.prefetch_queue.get(block=True)
      if batch is None:
        assert(self.one_pass)
        print('data reader: one pass finished')
        raise StopIteration()
      yield batch

def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle,
                  one_pass, params):
  num_samples = len(imdb['data'])
  batch_size = params['batch_size']

  n_sample = 0
  fetch_order = np.arange(num_samples)
  while True:
    # Shuffle the sample order for every epoch
    if n_sample == 0 and shuffle:
      fetch_order = np.random.permutation(num_samples)

    # Load batch from file
    # note that len(sample_ids) <= batch_size, not necessarily equal
    sample_ids = fetch_order[n_sample:n_sample+batch_size]
    batch = batch_loader.load_one_batch(sample_ids)
    prefetch_queue.put(batch, block=True)

    n_sample += len(sample_ids)
    if n_sample >= num_samples:
      # Put in a None batch to indicate a whole pass is over
      if one_pass:
        prefetch_queue.put(None, block=True)
      n_sample = 0
