"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Main class to execute programs using tensorflow fold loom API.

Program execution for explicit visual coreference resolution model in visual
dialog using neural module networks. Uses low-level loom API in tensorflow
fold:
https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/py/loom.md
for dynamic creation and execution of computation graphs.

Author: Satwik Kottur
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from tensorflow_fold.public import loom

import models_mnist.modules as lm
from models_mnist.assembler import INVALID_EXPR, _module_output_type


class ProgramExecutor:
  def __init__(self, inputs, output_pool, assembler, params) :
    """Initialize program execution subcomponent.

    Args:
      inputs:
      output_pool:
      assembler:
      params:
    """

    self.params = params
    # assembler dynamically assembles the graph at run time
    self._assembler = assembler
    #--------------------------------------------------------------------------

    # A. Create loom data inputs
    loom_inputs, used_inputs = self._build_loom_inputs(inputs, output_pool)

    # B. Create loom data types
    types = self._build_loom_types()
    self._loom_types = types

    # C. Create loom operations
    loom_ops_dict = self._build_loom_ops()
    self._loom_ops = loom_ops_dict

    # create a loom object
    keys = ['text', 'image', 'fact', 'time', 'round', 'text_feat']
    batch_ins = {types[k]: loom_inputs[k] for k in keys if k in loom_inputs}
    self._loom = loom.Loom(batch_inputs=batch_ins, named_ops=loom_ops_dict)

    # setup the inputs and outputs
    self.outputs = {'context': self.get_loom_output(),
                    'att': self.get_loom_output(types['attention']),
                    'logits': self.get_loom_output(types['float'])}

    # add invalid prog to used inputs
    used_inputs.extend(['prog_validity'])
    self.inputs = {ii: inputs[ii] for ii in used_inputs}
    # time/round place holder
    self.inputs['time'] = loom_inputs['time']
    self.inputs['round'] = loom_inputs['round']

  def create_weaver(self):
    """Creates a weaver object within the current loom object.
    """
    return self._loom.make_weaver()

  def get_loom_output(self, type_shape=None):
    """Return the loom output given the type and shape.
    """
    # default output is the context vector
    if type_shape is None:
      type_shape = self._loom_types['context']

    return self._loom.output_tensor(type_shape)

  #---------------------------------------------------------
  def _adjust_text(self, text):
    """
      takes text attention output from generator
      modifies it to have certain dimensions
    """

    params = self.params
    # transpose text to have batch first dimension
    text_mod = tf.transpose(text, [1, 0, 2])
    # split across rounds
    shape = text_mod.shape.as_list()
    new_size = [-1, params['num_rounds'], shape[1], shape[2]]
    return tf.reshape(text_mod, new_size)

  def _build_image_feature_network(self, image):
    """
      Takes in images and build features for the program
    """

    output = image
    # local aliases
    BN = tf.contrib.layers.batch_norm
    max_pool = tf.layers.max_pooling2d
    # Four convolutions networks followed by pooling
    for ii in range(2):
      # Convolutional Layer
      if self.params['align_image_features']:
        output = tf.layers.conv2d(inputs=output, filters=32,
                                  kernel_size=[5, 5], padding="valid",
                                  activation=None)
      else:
        output = tf.layers.conv2d(inputs=output, filters=32,
                                  kernel_size=[3, 3], padding="same",
                                  activation=None)

      # if batch norm is to be used
      if self.params['use_batch_norm']:
        output = BN(output, center=True, scale=True,
                    is_training=self.params['train_mode'])

      # Re_lU
      output = tf.nn.relu(output, 'relu')

      # Pooling Layer
      output = max_pool(output, pool_size=[2, 2], strides=2)

    for ii in range(2):
      # Convolutional Layer
      if self.params['align_image_features']:
        padding = 'valid'
      else:
        padding = 'same'

      output = tf.layers.conv2d(inputs=output, filters=64,
                                kernel_size=[3, 3], padding=padding,
                                activation=None)

      # if batch norm is to be used
      if self.params['use_batch_norm']:
        output = BN(output, center=True, scale=True,
                    is_training=self.params['train_mode'])

      # Re_lU
      output = tf.nn.relu(output, 'relu')

      # Pooling Layer
      output = max_pool(output, pool_size=[2, 2], strides=2)

    return output

  def _build_fact_encoder(self, inputs):
    """
    """

    # local alias
    params = self.params

    with tf.variable_scope(self.params['embed_scope'], reuse=True):
      size = [params['text_vocab_size'], params['text_embed_size']]
      embed_mat = tf.get_variable('embed_mat')

    # flatten
    # embed the words
    output = tf.nn.embedding_lookup(embed_mat, inputs['fact'])

    # pass through encoder
    cell = tf.contrib.rnn.BasicLSTMCell(params['text_embed_size'])

    # begin decoding
    for ii in range(0, params['num_layers']):
      # dynamic rnn
      output, states = tf.nn.dynamic_rnn(cell, output,
                                         sequence_length=inputs['fact_len'],
                                         dtype=tf.float32,
                                         scope='fact_layer_%d' % ii)

    # split roundwise
    fact_embed = states[1]
    text_dim = fact_embed.shape.as_list()[-1]
    fact_embed = tf.reshape(fact_embed, [-1, params['num_rounds'], text_dim])
    return fact_embed

  def _build_align_network(self, align_vec, align_gt):
    """
      Takes the caption alignment vector in and produces a binary
      classifier
    """

    params = self.params
    with tf.variable_scope('cap_align'):
      # construct an mlp on top to a binary classification
      align_vec = tf.contrib.layers.fully_connected(align_vec,
                          params['lstm_size']//2)
      align_vec = tf.contrib.layers.fully_connected(align_vec, 2,
                            activation_fn=None)

      # alias for criterion
      criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
      align_loss = criterion(logits=align_vec, labels=align_gt)
      align_loss = tf.reduce_mean(align_loss)

    return align_loss

  def _build_loom_inputs(self, inputs, output_pool):
    '''
      Sub routine to build the inputs to loom
    '''
    # --------- grab required inputs -------------
    loom_inputs = {}
    params = self.params

    # A. image
    # build image feature network
    image_feat = self._build_image_feature_network(inputs['image'])
    loom_inputs['image'], _ = lm.add_spatial_coord_map(image_feat)
    used_inputs = ['image']

    # B. text -- both question and caption
    key = 'ques_attended'
    if params['train_mode']: text = output_pool[key]
    else:
      text = inputs[key]
      used_inputs.append(key)
    adjusted_text = self._adjust_text(text)
    loom_inputs['text'] = adjusted_text
    batch_size = tf.shape(adjusted_text)[0]

    # C. Facts
    if params['use_fact']:
      loom_inputs['fact'] = self._build_fact_encoder(inputs)
      used_inputs.extend(['fact', 'fact_len'])

    concat_list = [adjusted_text]
    loom_inputs['text_feat'] = tf.concat(concat_list, -1)

    # D. time steps (internal placeholder)
    loom_inputs['time'] = tf.placeholder(tf.int32, (None, 1), 'time')
    loom_inputs['round'] = tf.placeholder(tf.int32, (None, 1), 'round')

    return loom_inputs, used_inputs

  def _build_loom_types(self):
    """Method to build loom types for given setting.
    """

    params = self.params
    encode_size = params['lstm_size']

    # create and save loom types
    types = {}
    types['time'] = loom.TypeShape('int32', (1,), 'time')
    types['round'] = loom.TypeShape('int32', (1,), 'round')
    types['float'] = loom.TypeShape('float32', (1,))
    types['context'] = loom.TypeShape('float32', (encode_size,), 'context')
    types['align'] = loom.TypeShape('float32', (encode_size,), 'align')

    size = (params['num_rounds'], params['text_embed_size'])
    types['fact'] = loom.TypeShape('float32', size, 'fact')
    size = (params['num_rounds'], params['max_dec_len'],
            params['text_embed_size'])
    types['text'] = loom.TypeShape('float32', size, 'text')

    size = (params['text_embed_size'],)
    types['text_slice'] = loom.TypeShape('float32', size, 'text_slice')

    # this depends on whether we want all features
    concat_dim = params['text_embed_size']

    size = (params['num_rounds'], params['max_dec_len'], concat_dim)
    types['text_feat'] = loom.TypeShape('float32', size, 'text_feat')
    size = (concat_dim,)
    types['text_feat_slice'] = loom.TypeShape('float32', size, 'text_feat_slice')

    # TODO: cleaner way to include spatial dimensions for img_feat
    size = (params['h_feat'], params['w_feat'], params['d_feat'] + 2)
    types['image'] = loom.TypeShape('float32', size, 'image')

    size = (params['h_feat'], params['w_feat'], 1)
    types['attention'] = loom.TypeShape('float32', size, 'att')

    return types

  def _build_loom_ops(self):
    """TODO(satwik): Some helper text here
    """

    params = self.params
    types = self._loom_types
    # create all modules under the same scope
    wt = params.get('priority_weight', 1.0)
    op_params = {'map_dim': params['map_size'], 'priority_weight': wt}
    with tf.variable_scope('loom_modules') as module_scope:
      op_params['module_scope'] = module_scope

    # creating ops
    loom_ops_dict = {}

    in_types = [types['float'], types['float']]
    out_types = [types['float']]
    loom_ops_dict['add'] = lm.BinaryLoomOp(in_types, out_types, tf.add)
    loom_ops_dict['divide'] = lm.BinaryLoomOp(in_types, out_types, tf.divide)
    in_types = [types['float']]
    loom_ops_dict['exp'] = lm.UnaryLoomOp(in_types, out_types, tf.exp)

    in_types = [types['attention'], types['attention']]
    out_types = [types['attention']]
    loom_ops_dict['add_attention'] = lm.BinaryLoomOp(in_types, out_types, tf.add)

    in_types = [types['attention'], types['attention']]
    out_types = [types['attention']]
    loom_ops_dict['max_attention'] = lm.BinaryLoomOp(in_types, out_types,
                                                     tf.maximum)

    # basic attention manipulation ops
    in_types = [types['attention'], types['float']]
    out_types = [types['attention']]
    loom_ops_dict['weight_attention'] = lm.AttentionWeightLoomOp(in_types,
                                                                 out_types)

    in_types = [types['text_feat_slice'], types['text_feat_slice'],
                types['round'], types['round']]
    out_types = [types['float']]
    op_params['amalgam_text_feats'] = params['amalgam_text_feats']
    op_params['text_embed_size'] = params['text_embed_size']
    loom_ops_dict['align_text'] = lm.AlignTextLoomOp(in_types, out_types, op_params)

    # slicing ops
    in_types = [types['text'], types['round'], types['time']]
    out_types = [types['text_slice']]
    loom_ops_dict['slice_text'] = lm.SliceTextLoomOp(in_types, out_types)

    in_types = [types['text_feat'], types['round'], types['time']]
    out_types = [types['text_feat_slice']]
    loom_ops_dict['slice_text_feat'] = lm.SliceTextLoomOp(in_types, out_types)

    # slice_answer_embedding
    in_types = [types['fact'], types['round']]
    out_types = [types['text_feat_slice']]
    loom_ops_dict['slice_fact'] = lm.SliceAnswerLoomOp(in_types, out_types)

    # normalize and complement
    in_types = [types['attention']]
    out_types = [types['attention']]
    loom_ops_dict['normalize_exclude']= lm.NormalizeExcludeLoomOp(in_types,
                                                                  out_types)

    #------------------------------------------------------------------
    # find module
    in_types = [types['image'], types['text_slice']]
    out_types = [types['attention']]
    loom_ops_dict['find'] = lm.FindLoomOp(in_types, out_types, op_params)

    # and module
    in_types = [types['attention'], types['attention']]
    loom_ops_dict['and_op'] = lm.AndLoomOp(in_types, out_types, op_params)
    # diff module
    loom_ops_dict['diff_op'] = lm.DiffLoomOp(in_types, out_types, op_params)

    # transform module
    in_types = [types['attention'], types['image'], types['text_slice']]
    loom_ops_dict['transform'] = lm.TransformLoomOp(in_types, out_types, op_params)

    # describe module
    out_types = [types['context']]
    op_params['encode_size'] = params['lstm_size']
    loom_ops_dict['describe'] = lm.DescribeLoomOp(in_types, out_types, op_params)

    # exist module
    loom_ops_dict['exist'] = lm.ExistLoomOp(in_types, out_types, op_params)

    # count module
    loom_ops_dict['count'] = lm.CountLoomOp(in_types, out_types, op_params)

    # invalid Module
    in_types = [types['image']]
    loom_ops_dict['invalid'] = lm.InvalidLoomOp(in_types, out_types, op_params)

    return loom_ops_dict
  #---------------------------------------------------------

  # setters and getters
  def get_outputs(self): return self.outputs
  def get_inputs(self): return self.inputs
  #------------------------------------------------------------

  # produce feed dict
  def produce_feed_dict(self, batch, output_pool=None, visualize=False):
    if 'prog' not in self.params['model']: return

    # dynamically assemble the graph, based on predicted tokens
    if self.params['train_mode']:
      ques_programs = batch['gt_layout']
    else:
      ques_programs = output_pool['pred_tokens']
    tokens = {'ques': ques_programs}
    weaver, loom_outputs, invalid_prog \
            = self._assembler.assemble(tokens, self, visualize)
    # build feed dict from loom
    feed_dict = weaver.build_feed_dict(loom_outputs)

    # feed invalid Prog
    feed_dict[self.inputs['prog_validity']] = np.array(invalid_prog['ques'])

    # additional feeds
    feed_dict[self.inputs['image']] = batch['imgs']

    max_time = self.params['max_dec_len']
    feed_dict[self.inputs['time']] = np.arange(max_time).reshape([-1, 1])
    round_ranges = np.arange(self.params['num_rounds']).reshape([-1, 1])
    feed_dict[self.inputs['round']] = round_ranges

    # fact is needed
    if self.params['use_fact']:
      feed_dict[self.inputs['fact']] = batch['fact']
      feed_dict[self.inputs['fact_len']] = batch['fact_len']

    if not self.params['train_mode']:
      # list of labels to read from output pool conditionally
      labels = ['ques_attended', 'cap_attended', 'ques_enc', 'cap_enc']
      for label in labels:
        if label in self.inputs:
          feed_dict[self.inputs[label]] = output_pool[label]

    return feed_dict
  #------------------------------------------------------------

  # segregating the outputs
  def segregrate_outputs(self, output):
    '''
      Go over the outputs, cap tokens and ques tokens
    '''
    ques_tokens = output['pred_tokens']
    mod_out_type = _module_output_type
    mod_dict = self._assembler.module_names

    att = output['att']
    weights = output['weight']

    # segregrated outputs
    sep_att = []
    sep_wts = []
    wt_labels = []
    num_reuse = 0 att_ind = 0 weight_ind = 0
    # assume a batch size of 1
    for r_id in range(self.params['num_rounds']):
      #refer_seen = False
      for t_id in range(self.params['max_dec_length']):
        cur_module = mod_dict[ques_tokens[t_id, r_id]]
        if cur_module == '<eos>':
          # even answer has a weight now
          if self.params['use_answer'] or self.params['use_fact']:
            wt_labels.append('A%d' % r_id)
            num_reuse += 1
          break

        if mod_out_type[cur_module] == 'att':
          sep_att.append(('ques', t_id, r_id, att[att_ind]))
          att_ind += 1

        if cur_module == '_Refer':
          refer_seen = True
          st = weight_ind
          end = weight_ind + num_reuse
          sep_wts.append((r_id, weights[st:end], wt_labels))
          weight_ind += num_reuse

        '''
        if self.params['reuse_refer'] and cur_module == '_Refer':
          wt_labels.append('Q%d_%d' % (r_id, t_id))
          num_reuse += 1

        if cur_module == '_Find':
          if refer_seen and self.params['remove_aux_find']: continue
          wt_labels.append('Q%d_%d' % (r_id, t_id))
          num_reuse += 1
        '''

    for arg in sep_wts: assert(abs(np.sum(arg[1]) - 1.0) < 1e-5)

    assert(weight_ind == weights.shape[0])
    #assert(att_ind == att.shape[0])

    return sep_att, sep_wts
