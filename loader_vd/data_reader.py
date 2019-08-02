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


class BatchLoaderVD:
  """Subclass to DataReader that serves batches during training.
  """

  # adjust for current directory
  def _adjust_image_dir(self, path):
    # split before data, and append with pwd
    return os.path.join(os.getcwd(), 'data', path.split('data/')[-1])

  def __init__(self, imdb, params):
    """Initialize by reading the data and pre-processing it.
    """

    self.imdb = imdb
    self.params = params
    self.fetch_options = self.params.get('fetch_options', False)
    self.preload_features = params['preload_features']
    self.num_inst = len(self.imdb['data'])
    self.num_rounds = len(self.imdb['data'][0]['question_ind'])

    # check if vgg features are to be used
    self.use_vgg = 'vgg' in self.params['feature_path']

    # load vocabulary
    vocab_path = params['text_vocab_path']
    self.vocab_dict = text_processing.VocabDict(vocab_path)
    self.T_encoder = params['max_enc_len']

    # record special token ids
    self.start_token_id = self.vocab_dict.word2idx('<start>')
    self.end_token_id = self.vocab_dict.word2idx('<end>')
    self.pad_token_id = self.vocab_dict.word2idx('<pad>')

    # peek one example to see whether answer and gt_layout are in the data
    test_data = self.imdb['data'][0]
    self.load_gt_layout = test_data.get('gt_layout_tokens', False)
    if 'load_gt_layout' in params:
      self.load_gt_layout = params['load_gt_layout']

    # decide whether or not to load gt textatt
    self.supervise_attention = params['supervise_attention']
    self.T_decoder = params['max_dec_len']
    self.assembler = params['assembler']

    # load one feature map to peek its size
    feats = np.load(self._adjust_image_dir(test_data['feature_path']))
    self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]

    # convert to tokens
    self.digitizer = lambda x: [self.vocab_dict.word2idx(w) for w in x]

    if 'prog' in self.params['model']:
      # preload features
      if self.preload_features:
        img_paths = set([ii['feature_path'] for ii in self.imdb['data']])
        self.img_feats = {ii:np.load(ii) for ii in progressbar(img_paths)}

      # if VGG is to be used
      if self.use_vgg:
        # inform the dataloader to use self.img_feats
        self.preload_features = True
        img_paths = set([ii['feature_path'] for ii in self.imdb['data']])

        # first read the index file
        index_file = os.path.join(self.params['input_img'], 'img_id.json')
        with open(index_file, 'r') as file_id:
          index_data = json.load(file_id)

        # get the split -- either train / val
        for ii in img_paths: break
        split = ii.split('/')[-2][:-4]

        # read the features for that particular split
        self.img_index = {img_id: index for index, img_id
                          in enumerate(index_data[split])}
        feature_file = os.path.join(self.params['input_img'],
                                    'data_img_%s.h5' % split)
        key = 'images_test' if split == 'val' else 'images_train'
        self.img_feats = h5py.File(feature_file)[key]

        # check if all the images in img_paths are in img_index
        count = 0
        for ii in img_paths:
          img_id = '/'.join(ii.split('/')[-2:])
          if img_id.replace('npy', 'jpg') not in self.img_index:
            count += 1
        print('Missing: %d image features' % count)

        # adjust the feature sizes
        self.feat_H, self.feat_W, self.feat_D = self.img_feats.shape[1:]
        self.zero_feature = np.zeros((1,) + self.img_feats.shape[1:])

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
    max_len = self.T_encoder # question + answer appended
    num_rounds = len(self.imdb['data'][0]['question_ind'])
    fact = np.zeros((num_diags, num_rounds, max_len))
    fact_len = np.zeros((num_diags, num_rounds))
    fact.fill(self.pad_token_id)

    for diag_id, datum in enumerate(self.imdb['data']):
      for r_id in range(num_rounds - 1):
        q_id = datum['question_ind'][r_id]
        a_id = datum['answer_ind'][r_id]
        ques, q_len = self.imdb['ques'][q_id], self.imdb['ques_len'][q_id]
        ans, a_len = self.imdb['ans'][a_id], self.imdb['ans_len'][a_id]

        # handle overflow
        bound = min(q_len, max_len)
        fact[diag_id, r_id, :bound] = ques[:bound]
        if bound < max_len:
          bound = min(q_len + a_len, max_len)
          fact[diag_id, r_id, q_len:bound] = ans[:bound-q_len]
        fact_len[diag_id, r_id] = bound

    # flatten
    self.imdb['fact'] = fact
    self.imdb['fact_len'] = fact_len
  #--------------------------------------------------------------------------

  def _construct_history(self):
    """Method to construct history, which concatenates entire dialogs so far.
    """

    print('Constructing history..')

    num_diags = len(self.imdb['data'])
    max_len = self.T_encoder * 2 # question + answer appended
    num_rounds = len(self.imdb['data'][0]['question_ind'])
    history = np.zeros((num_diags, num_rounds, max_len))
    hist_len = np.zeros((num_diags, num_rounds))
    history.fill(self.pad_token_id)

    for diag_id, datum in enumerate(self.imdb['data']):
      # history for first round is caption
      c_id = datum['caption_ind']
      cap_len = self.imdb['cap_len'][c_id]
      caption = self.imdb['cap'][c_id]

      # handle overflow
      bound = min(cap_len, max_len)
      hist_len[diag_id, 0] = bound
      history[diag_id, 0, :bound] = caption[:bound]

      for r_id in range(num_rounds - 1):
        q_id = datum['question_ind'][r_id]
        a_id = datum['answer_ind'][r_id]
        ques, q_len = self.imdb['ques'][q_id], self.imdb['ques_len'][q_id]
        ans, a_len = self.imdb['ans'][a_id], self.imdb['ans_len'][a_id]

        # handle overflow
        bound = min(q_len, max_len)
        history[diag_id, r_id + 1, :bound] = ques[:bound]
        if bound < max_len:
          bound = min(q_len + a_len, max_len)
          history[diag_id, r_id + 1, q_len:bound] = ans[:bound-q_len]
        hist_len[diag_id, r_id + 1] = bound

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

    # whether to flatten or not
    flatten = 'dial' not in self.params['model']
    flatten = 'nmn-cap' not in self.params['model']
    num_rounds = self.num_rounds

    # captions
    if flatten:
      cap_inds = [self.imdb['data'][ii]['caption_ind'] for ii in sample_ids
                  for _ in range(num_rounds)]
    else:
      cap_inds = [self.imdb['data'][ii]['caption_ind'] for ii in sample_ids]
    cap_batch = self.imdb['cap'][cap_inds][:, :self.T_encoder]
    cap_len = self.imdb['cap_len'][cap_inds]

    # get caption programs
    cap_prog = None
    cap_gt_att = None
    if 'nmn-cap' in self.params['model']:
      cap_prog = np.zeros((self.T_decoder, len(cap_inds)), np.int32)
      cap_prog.fill(eos_token)
      for spot, ii in enumerate(cap_inds):
        layout = self.imdb['cap_prog'][ii]

        cap_prog[:, spot] = \
          self.assembler.module_list2tokens(layout, self.T_decoder)

        # also get attention for supervision
        if self.supervise_attention:
          cap_gt_att = np.zeros((self.T_decoder, self.T_encoder, \
                      actual_batch_size, 1), np.float32)

          for spot, ii in enumerate(cap_inds):
            for t_id, att in enumerate(self.imdb['cap_prog_att'][ii]):
              span = att[1] - att[0]
              # NOTE: number of attention hardwired to be <= 4
              if span > 0 or span == 0: continue
              if span == 0: continue
              cap_gt_att[t_id, att[0]:att[1], spot] = 1/span

    # questions
    ques_inds = [jj for ii in sample_ids
                 for jj in self.imdb['data'][ii]['question_ind']]
    ques_batch = self.imdb['ques'][ques_inds][:, :self.T_encoder].transpose()
    ques_len = self.imdb['ques_len'][ques_inds]
    ques_ids = [jj for ii in sample_ids
                for jj in self.imdb['data'][ii]['question_id']]
    gt_index = [jj for ii in sample_ids
                for jj in self.imdb['data'][ii]['gt_ind']]

    # answers
    ans_inds = [jj for ii in sample_ids
                for jj in self.imdb['data'][ii]['answer_ind']]

    ans_batch_in = self.imdb['ans_in'][ans_inds][:, :self.T_encoder]
    ans_batch_out = self.imdb['ans_out'][ans_inds][:, :self.T_encoder]
    ans_batch = self.imdb['ans_in'][ans_inds][:, 1:self.T_encoder]
    ans_len = self.imdb['ans_len'][ans_inds]

    # getting history
    if self.use_history:
      history = self.imdb['hist'][sample_ids]
      hist_len = self.imdb['hist_len'][sample_ids]
    else:
      history, hist_len = None, None

    # image features
    if 'prog' in self.params['model']:
      # single copy per conversation
      image_feats = np.zeros((actual_batch_size, self.feat_H,
                              self.feat_W, self.feat_D), np.float32)

    else:
      image_feats = None

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

    gt_attention = None
    if self.supervise_attention:
      gt_attention = np.zeros((self.T_decoder, self.T_encoder,
                               num_rounds * actual_batch_size, 1), np.float32)

    # mask for weights, for history attention
    weight_mask = []
    for n in range(len(sample_ids)):
      iminfo = self.imdb['data'][sample_ids[n]]

      # image features
      if 'prog' in self.params['model']:
        # if VGG features are to be used
        if self.use_vgg:
          img_id = '/'.join(iminfo['feature_path'].split('/')[-2:])
          img_id = img_id.replace('npy', 'jpg')
          if img_id in self.img_index:
            f_ind = self.img_index[img_id]
            cur_feat = self.img_feats[f_ind]
          else:
            cur_feat = self.zero_feature

        else:
          # use preloaded image features
          feat_path = self._adjust_image_dir(iminfo['feature_path'])
          if not self.preload_features: cur_feat = np.load(feat_path)
          else: cur_feat = self.img_feats[feat_path]

        # single copy per conversation
        image_feats[n] = cur_feat

      image_path[n] = iminfo['image_path']

      # programs
      if self.load_gt_layout:
        # go over all the questions
        for r_id, layout in enumerate(iminfo['gt_layout_tokens']):
          gt_layout_batch[:, num_rounds * n + r_id] = \
          self.assembler.module_list2tokens(layout, self.T_decoder)

      if self.supervise_attention:
        num_refers = 0
        for r_id, att in enumerate(iminfo['gt_layout_att']):
          for t_id in range(att.shape[0]):
            index = num_rounds * n + r_id
            span = att[t_id, 1] - att[t_id, 0]
            # NOTE: number of attention timesteps hardwired to be <= 4
            if span > 4 or span == 0: continue
            gt_attention[t_id, att[t_id,0]:att[t_id,1], index] = 1/span

      # if options are not needed, continue
      if not self.fetch_options: continue
      #------------------------------------------------------------------

      # get options
      opt_inds = [jj for ii in sample_ids
              for jj in self.imdb['data'][ii]['option_ind']]
      num_options = len(opt_inds[0])
      opt_batch_in = [None] * num_options
      opt_batch_out = [None] * num_options
      opt_len = [None] * num_options
      for ii in range(num_options):
        cur_inds = [jj[ii] for jj in opt_inds]
        opt_batch_in[ii] = self.imdb['ans_in'][cur_inds][:, :self.T_encoder]
        opt_batch_out[ii] = self.imdb['ans_out'][cur_inds][:, :self.T_encoder]
        opt_len[ii] = self.imdb['ans_len'][cur_inds]
      #------------------------------------------------------------------

    batch = {'ques': ques_batch, 'ques_len': ques_len,
             'ques_id': ques_ids, 'gt_layout': gt_layout_batch,
             'gt_att' : gt_attention,
             'cap': cap_batch, 'cap_len': cap_len, 'cap_prog': cap_prog,
             'cap_att': cap_gt_att,
             'hist': history, 'hist_len': hist_len, 'ans_in': ans_batch_in,
             'ans_out': ans_batch_out, 'ans_len':ans_len, 'ans': ans_batch,
             'fact': fact, 'fact_len': fact_len,
             'img_feat': image_feats, 'img_path': image_path}

    #------------------------------------------------------------------
    # further add options
    if self.fetch_options:
      options = {'opt_in': opt_batch_in, 'opt_out': opt_batch_out,\
                 'opt_len': opt_len, 'gt_ind': gt_index}
      batch.update(options)
    #------------------------------------------------------------------
    if 'nmn-cap' not in self.params['model']:
      return batch

    # getting data for training alignment on caption
    if actual_batch_size > 1:
      info = [batch['cap'], batch['cap_len'],
            batch['cap_prog'].transpose()]
      if batch['cap_att'] is not None:
        info.append(batch['cap_att'].transpose((2, 0, 1, 3)))

      shuffled = support.shuffle(info, actual_batch_size)

      batch['sh_cap'], batch['sh_cap_len'] = shuffled[:2]
      batch['sh_cap_prog'] = shuffled[2].transpose()
      batch['align_gt'] = np.ones(num_rounds*actual_batch_size).astype('int32')

      if batch['cap_att'] is not None:
        batch['sh_cap_att'] = shuffled[3].transpose((1, 2, 0, 3))
      for ii in range(actual_batch_size):
        start =  num_rounds * ii + num_rounds // 2
        end = num_rounds * (ii+1)
        batch['align_gt'][start:end] = 0
    else:
      batch['sh_cap'] = np.tile(batch['cap'], [num_rounds, 1])
      batch['sh_cap_len'] = np.tile(batch['cap_len'], [num_rounds])
      batch['sh_cap_prog'] = np.tile(batch['cap_prog'], [1, num_rounds])
      batch['sh_cap_att'] = np.tile(batch['cap_att'], [1, 1, num_rounds, 1])
      batch['align_gt'] = np.ones(num_rounds*actual_batch_size).astype('int32')

    return batch


class DataReader:
  """Main dataloader class for experiments on Visual Dialog.
  """

  def __init__(self, params):
    imdb_path = params['path']
    print('Loading imdb from: %s' % imdb_path)
    if imdb_path.endswith('.npy'): imdb = np.load(imdb_path)
    else: raise Type_error('unknown imdb format.')
    self.imdb = imdb[()]

    self.shuffle = params.get('shuffle', True)
    self.one_pass = params.get('one_pass', False)
    self.prefetch_num = params.get('num_prefetch', 8)
    self.params = params
    copy_args = {'max_enc_len', 'max_dec_len', 'text_vocab_path', 'model',
                 'batch_size', 'use_fact', 'preload_features',
                 'supervise_attention','generator', 'feature_path'}
    self.params.update({ii: params['args'][ii] for ii in copy_args
                        if ii in params['args'] and
                        params['args'][ii] is not None})

    # VD data loader
    self.batch_loader = BatchLoaderVD(self.imdb, self.params)

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
