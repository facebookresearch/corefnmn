"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Main class to generate programs for questions and captions.

Program generator for explicit visual coreference resolution model in visual
dialog using neural module networks, called CorefNMN.
This subcomponent uses memory network augmentation to figure out if an entity
has been seen before and/or if it needs resolution using history.

Author: Satwik Kottur
"""

import numpy as np
import tensorflow as tf
from models_mnist.generator_attnet import AttSeq2Seq
from util import support

# alias
linear = tf.contrib.layers.fully_connected

# behavior based on type of model
class ProgramGenerator:
  def __init__(self, inputs, assembler, params):
    """Initialize program generator.

    Args:
      inputs:
      assembler:
      params:
    """

    self.params = params
    outputs = {}
    used_inputs = []

    # create embedding matrix
    with tf.variable_scope('embed', reuse=None) as embed_scope:
      size = [params['text_vocab_size'], params['text_embed_size']]
      embed_mat = tf.get_variable('embed_mat', size)

    # remember the scope for further use
    params['embed_scope'] = embed_scope

    cell = tf.contrib.rnn.BasicLSTMCell(params['lstm_size'])
    #--------------------------------------------------------

    # if program is to be predicted
    if 'prog' in params['model']:
      # define a constant for internal use
      use_gt_prog = tf.constant(params['use_gt_prog'], dtype=tf.bool)

      # use a low level model and construct internals
      self.rnn = AttSeq2Seq(inputs, use_gt_prog, assembler, params)
      # if memory based generator is used
      if params['generator'] == 'mem':
        used_inputs.extend(['hist', 'hist_len'])

      outputs['encoder_output'] = self.rnn.encoder_outputs
      outputs['pred_tokens'] = self.rnn.predicted_tokens
      outputs['neg_entropy'] = tf.reduce_mean(self.rnn.neg_entropy)

      # check if attHistory exists
      if hasattr(self.rnn, 'att_history'):
        outputs['att_history'] = self.rnn.att_history

      # also add the encoder states (based on the flag)
      concat_list = [ii.h for ii in self.rnn.encoder_states]
      outputs['enc_dec_h'] = tf.stack(concat_list)
      concat_list = [ii.c for ii in self.rnn.encoder_states]
      outputs['enc_dec_c'] = tf.stack(concat_list)

      # alias
      attention = self.rnn.atts

      # compute attended questions here
      # word_vec has shape [T_decoder, N, 1]
      word_vecs = tf.reduce_sum(attention * self.rnn.embedded_input_seq, axis=1)
      size = [params['max_dec_len'], None, params['text_embed_size']]
      word_vecs.set_shape(size)
      outputs['attention'] = attention
      outputs['ques_attended'] = word_vecs
      #outputs['ques_attended'] = self.rnn.word_vecs

      # log probability of each generated sequence
      outputs['log_seq_prob'] = tf.reduce_sum(
                                  tf.log(self.rnn.token_probs + 1e-10), axis=0)
      outputs['ques_prog_loss'] = tf.reduce_mean(-outputs['log_seq_prob'])
      q_output = tf.transpose(self.rnn.encoder_outputs, perm=[1, 0, 2])
      q_output = support.last_relevant(q_output, inputs['ques_len'])
      # bloat the first two dimensions
      q_output = tf.expand_dims(q_output, axis=0)
      outputs['ques_enc'] = tf.expand_dims(q_output, axis=0)

      # keep track of inputs actually used
      used_inputs.extend(['ques', 'ques_len', 'prog_gt'])
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    # setup the inputs and outputs
    # should have at least one loss
    total_loss  = outputs.get('ques_prog_loss', tf.constant(0.0))
    outputs['prog_pred_loss'] = outputs['ques_prog_loss']
    self.outputs = outputs
    self.inputs = {ii: inputs[ii] for ii in used_inputs}
  #------------------------------------------------------------

  # setters and getters
  def get_outputs(self):
    return self.outputs

  def get_inputs(self):
    return self.inputs
  #------------------------------------------------------------

  # produce feed dict
  def produce_feed_dict(self, batch, prev_output=None):
    feed_dict = {}
    feed_dict[self.inputs['ques']] = batch['ques']
    feed_dict[self.inputs['ques_len']] = batch['ques_len']

    # add program
    if 'prog' in self.params['model']:
      feed_dict[self.inputs['prog_gt']] = batch['gt_layout']

    # add history
    if self.params['generator'] == 'mem':
      feed_dict[self.inputs['hist']] = batch['hist']
      feed_dict[self.inputs['hist_len']] = batch['hist_len']

    return feed_dict
