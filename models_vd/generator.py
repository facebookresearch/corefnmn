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
from models_vd.generator_attnet import AttSeq2Seq
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

      # if attention is to be supervised
      if params['supervise_attention']:
        # get mask out of the program supervision
        mask = tf.cast(inputs['prog_att_gt'] > 0, tf.float32)
        used_inputs.append('prog_att_gt')

        # binary supervision loss
        sum_mask = tf.reduce_sum(mask, 1)
        sum_mask = tf.expand_dims(sum_mask, 1)
        sum_mask = tf.cast(sum_mask > 0, tf.float32)
        tile_size = (1, self.params['max_enc_len'], 1, 1)
        tile_mask = tf.tile(sum_mask, tile_size)
        num_tokens = tf.maximum(tf.reduce_sum(tile_mask), 1)
        # stop gradients
        num_tokens = tf.stop_gradient(num_tokens)
        tile_mask = tf.stop_gradient(tile_mask)

        criterion = tf.nn.sigmoid_cross_entropy_with_logits
        att_loss = criterion(labels=mask,logits=attention)
        att_loss = tf.reduce_sum(tf.multiply(att_loss, tile_mask))
        att_loss = att_loss / num_tokens

        outputs['att_loss'] = att_loss

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

    # programs for captions
    if 'nmn-cap' in params['model']:
      # define a constant for internal use
      use_gt_prog = tf.constant(params['use_gt_prog'], dtype=tf.bool)

      # use a low level model and construct internals
      # pretend captions to be questions for code reusability
      fake_ins = {'ques': tf.transpose(inputs['cap'], perm=[1, 0]),
                  'ques_len': inputs['cap_len'],
                  'prog_gt': inputs['cap_prog_gt']}
      function_ins = [fake_ins, use_gt_prog, assembler, params]

      # if captions and questions share encoder
      # default value for sharing encoding
      self.params['share_encoder'] = self.params.get('share_encoder', False)
      if not self.params['share_encoder']:
        function_ins[0]['fake'] = True
      else:
        function_ins += [True]

      self.rnn_cap = AttSeq2Seq(*function_ins)
      used_inputs.extend(['cap', 'cap_len', 'cap_prog_gt'])

      outputs['pred_tokens_cap'] = self.rnn_cap.predicted_tokens
      outputs['neg_entropy_cap'] = tf.reduce_mean(self.rnn_cap.neg_entropy)
      #------------------------------------------------------------------
      # alias
      attention = self.rnn_cap.atts

      # if attention is to be supervised
      if params['supervise_attention']:
        # get mask out of the program supervision
        mask = tf.cast(inputs['cap_att_gt'] > 0, tf.float32)

        # binary supervision loss
        sum_mask = tf.reduce_sum(mask, 1)
        sum_mask = tf.expand_dims(sum_mask, 1)
        sum_mask = tf.cast(sum_mask > 0, tf.float32)
        tile_size = (1, self.params['max_enc_len'], 1, 1)
        tile_mask = tf.tile(sum_mask, tile_size)
        num_tokens = tf.maximum(tf.reduce_sum(tile_mask), 1)
        # stop gradients
        num_tokens = tf.stop_gradient(num_tokens)
        tile_mask = tf.stop_gradient(tile_mask)

        criterion = tf.nn.sigmoid_cross_entropy_with_logits
        att_loss = criterion(labels=mask,logits=attention)
        att_loss = tf.reduce_sum(tf.multiply(att_loss, tile_mask))
        att_loss_cap = att_loss / num_tokens

        # additional add the multiplier
        outputs['att_loss_cap'] = att_loss_cap
        used_inputs.append('cap_att_gt')

      # compute attended questions here
      # word_vec has shape [T_decoder, N, 1]
      word_vecs = tf.reduce_sum(attention * self.rnn_cap.embedded_input_seq,
                                    axis=1)
      size = [params['max_dec_len'], None, params['text_embed_size']]
      word_vecs.set_shape(size)
      outputs['attention_cap'] = attention
      outputs['cap_attended'] = word_vecs
      #outputs['cap_attended'] = self.rnn_cap.word_vecs
      #------------------------------------------------------------------

      # log probability of each generated sequence
      log_prob_cap_token = tf.log(self.rnn_cap.token_probs + 1e-10)
      outputs['log_seq_prob_cap'] = tf.reduce_sum(log_prob_cap_token, axis=0)
      outputs['cap_prog_loss'] = tf.reduce_mean(-outputs['log_seq_prob_cap'])

      c_output = tf.transpose(self.rnn_cap.encoder_outputs, perm=[1, 0, 2])
      c_output = support.last_relevant(c_output, inputs['cap_len'])
      # bloat the first two dimensions
      c_output = tf.expand_dims(c_output, axis=0)
      outputs['cap_enc'] = tf.expand_dims(c_output, axis=0)

      used_inputs.extend(['cap', 'cap_len'])
    #------------------------------------------------------------------

    # setup the inputs and outputs
    # should have at least one loss
    total_loss = (outputs.get('ques_prog_loss', tf.constant(0.0)) +
                  outputs.get('cap_prog_loss', tf.constant(0.0)) +
                  outputs.get('att_loss', tf.constant(0.0)) +
                  outputs.get('att_loss_cap', tf.constant(0.0)))
    outputs['prog_pred_loss'] = total_loss

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

    # attention for program
    if self.params['supervise_attention']:
      feed_dict[self.inputs['prog_att_gt']] = batch['gt_att']

    # add captions
    if 'cap' in self.params['model']:
      feed_dict[self.inputs['cap']] = batch['cap']
      feed_dict[self.inputs['cap_len']] = batch['cap_len']

    # add history
    if self.params['generator'] == 'mem':
      feed_dict[self.inputs['hist']] = batch['hist']
      feed_dict[self.inputs['hist_len']] = batch['hist_len']

    # nmn on captions
    if 'nmn-cap' in self.params['model']:
      feed_dict[self.inputs['cap']] = batch['sh_cap']
      feed_dict[self.inputs['cap_len']] = batch['sh_cap_len']
      feed_dict[self.inputs['cap_prog_gt']] = batch['sh_cap_prog']

      if self.params['supervise_attention']:
        feed_dict[self.inputs['cap_att_gt']] = batch['sh_cap_att']

    return feed_dict
  #------------------------------------------------------------
