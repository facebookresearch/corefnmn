"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

TODO(satwik): Add a reasonable description to the file.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.cnn import fc_layer as fc, conv_relu_layer as conv_relu
from tensorflow.contrib.layers import fully_connected as FC
from tensorflow.contrib.rnn import LSTMStateTuple
from util import support


def _get_valid_tokens(X, W, b):
  constraints_validity = tf.greater_equal(tf.tensordot(X, W, axes=1) - b, 0)
  token_validity = tf.reduce_all(constraints_validity, axis=2)
  return tf.stop_gradient(token_validity)
#------------------------------------------------------------------------------


def _update_decoding_state(X, s, P):
  X = X + tf.nn.embedding_lookup(P, s)  # X = X + S P
  return tf.stop_gradient(X)
#------------------------------------------------------------------------------


def _get_lstm_cell(num_layers, lstm_dim, apply_dropout):
  if isinstance(lstm_dim, list):  # Different layers have different dimensions
    if not len(lstm_dim) == num_layers:
      raise ValueError('the length of lstm_dim must be equal to num_layers')
    cell_list = []
    for l in range(num_layers):
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim[l], state_is_tuple=True)
      # Dropout is only applied on output of the 1st to second-last layer.
      # The output of the last layer has no dropout
      if apply_dropout and l < num_layers-1:
        dropout_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
          output_keep_prob=0.5)
      else:
        dropout_cell = lstm_cell
      cell_list.append(dropout_cell)
  else:  # All layers has the same dimension.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=True)
    # Dropout is only applied on output of the 1st to second-last layer.
    # The output of the last layer has no dropout
    if apply_dropout:
      dropout_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                            output_keep_prob=0.5)
    else:
      dropout_cell = lstm_cell
    cell_list = [dropout_cell] * (num_layers-1) + [lstm_cell]

  cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
  return cell
#------------------------------------------------------------------------------


# Sequence to Sequence with attention
class AttSeq2Seq:
  def __init__(self, holders, use_gt_prog, assembler, params, reuse=None):
    self.T_decoder = params['max_dec_len']
    self.encoder_num_vocab = params['text_vocab_size']
    self.encoder_embed_dim = params['text_embed_size']
    self.decoder_num_vocab = params['prog_vocab_size']
    self.decoder_embed_dim = params['prog_embed_size']
    self.lstm_dim = params['lstm_size']
    self.num_layers = params['num_layers']
    self.EOS_token = assembler.EOS_idx
    self.embed_scope = params['embed_scope']
    self.temperature = params.get('temperature', 1)

    # if word vectors need to be used or lstm outputs for attention
    params['use_word_vectors'] = 'wv-att' in params['model']
    params['generator'] = params.get('generator', 'ques')
    self.params = params

    # decoding transition variables
    self.P = to_T(assembler.P, dtype=tf.int32)
    self.W = to_T(assembler.W, dtype=tf.int32)
    self.b = to_T(assembler.b, dtype=tf.int32)

    self.encoder_dropout = params['enc_dropout']
    self.decoder_dropout = params['dec_dropout']
    self.decoder_sampling = params['dec_sampling']

    # detect fake inputs
    if 'fake' in holders: scope = 'enc_dec_cap'
    else: scope = 'enc_dec'
    with tf.variable_scope(scope, reuse=reuse):
      # build a special encoder, if needed
      if 'fake' not in holders and params['generator'] == 'mem':
        self._build_memory_encoder(holders)
      else:
        # build a normal encoder
        self._build_encoder(holders['ques'], holders['ques_len'])
      self._build_decoder(use_gt_prog, holders['prog_gt'])

  # build a usual encoder, ques based
  def _build_encoder(self, input_seq_batch, seq_len_batch, scope='encoder',
    reuse=None):
    lstm_dim = self.lstm_dim
    num_layers = self.num_layers
    apply_dropout = self.encoder_dropout

    with tf.variable_scope(scope, reuse=reuse):
      #T = tf.shape(input_seq_batch)[0]
      T = input_seq_batch.shape.as_list()[0]
      N = tf.shape(input_seq_batch)[1]
      self.T_encoder = T
      self.N = N
      with tf.variable_scope(self.embed_scope, reuse=True):
        embedding_mat = tf.get_variable('embed_mat', [self.encoder_num_vocab,
                                                      self.encoder_embed_dim])
      # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
      embedded_seq = tf.nn.embedding_lookup(embedding_mat, input_seq_batch)
      self.embedded_input_seq = embedded_seq

      # The RNN
      cell = _get_lstm_cell(num_layers, lstm_dim, apply_dropout)

      # encoder_outputs has shape [T, N, lstm_dim]
      encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell, embedded_seq,
                                                          seq_len_batch,
                                                          dtype=tf.float32,
                                                          time_major=True,
                                                          scope='lstm')
      self.encoder_outputs = encoder_outputs
      self.encoder_states = encoder_states

      # check if wv flag is set
      if self.params['use_word_vectors']:
        # transform the encoder outputs for further attention alignments
        # encoder_outputs_flat has shape [T, N, lstm_dim]
        encoder_h_transformed = fc('encoder_h_transform',
          tf.reshape(embedded_seq, [-1, self.encoder_embed_dim]),
                            output_dim=lstm_dim)
      else:
        # transform the encoder outputs for further attention alignments
        # encoder_outputs_flat has shape [T, N, lstm_dim]
        encoder_h_transformed = fc('encoder_h_transform',
          tf.reshape(encoder_outputs, [-1, lstm_dim]), output_dim=lstm_dim)

      encoder_h_transformed = tf.reshape(encoder_h_transformed,
                       to_T([T, N, lstm_dim]))
      self.encoder_h_transformed = encoder_h_transformed

      # seq_not_finished has shape [T, N, 1], where seq_not_finished[t, n]
      # is 1 iff sequence n is not finished at time t, and 0 otherwise
      seq_not_finished = tf.less(tf.range(T)[:, tf.newaxis, tf.newaxis],
                   seq_len_batch[:, tf.newaxis])
      seq_not_finished = tf.cast(seq_not_finished, tf.float32)
      self.seq_not_finished = seq_not_finished

  # build a special encoder
  def _build_memory_encoder(self, holders, scope='encoder', reuse=None):
    lstm_dim = self.lstm_dim
    num_layers = self.num_layers
    apply_dropout = self.encoder_dropout

    input_seq = holders['ques']
    input_seq_len = holders['ques_len']

    # facts/memories
    hist_size = holders['hist'].shape.as_list()
    hist_flat = tf.reshape(holders['hist'], [-1, hist_size[2]])
    hist_len_flat = tf.reshape(holders['hist_len'], [-1])

    with tf.variable_scope(scope, reuse=reuse):
      T = input_seq.shape.as_list()[0]
      N = tf.shape(input_seq)[1]
      self.T_encoder = T
      self.N = N
      with tf.variable_scope(self.embed_scope, reuse=True):
        embed_mat = tf.get_variable('embed_mat', [self.encoder_num_vocab,
                                                  self.encoder_embed_dim])
      # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
      embed_seq = tf.nn.embedding_lookup(embed_mat, input_seq)
      self.embedded_input_seq = embed_seq

      # The RNN
      cell = _get_lstm_cell(num_layers, lstm_dim, apply_dropout)

      # encoder_outputs has shape [T, N, lstm_dim]
      encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell,
                  embed_seq, input_seq_len, dtype=tf.float32,
                  time_major=True, scope='lstm')
      self.encoder_outputs = encoder_outputs

      # batch first encoder outputs
      batch_encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
      ques_enc = support.last_relevant(batch_encoder_outputs, input_seq_len)
      size = [-1, self.params['num_rounds'], self.params['lstm_size']]
      ques_enc = tf.reshape(ques_enc, size)
      self.encoder_states = encoder_states

      # similarly encode history
      hist_out = tf.nn.embedding_lookup(embed_mat, hist_flat)
      # rnns to encode history
      cell = tf.contrib.rnn.BasicLSTMCell(self.params['lstm_size'])
      for ii in range(0, self.params['num_layers']):
        # dynamic rnn
        hist_out, states = tf.nn.dynamic_rnn(cell, hist_out, \
                sequence_length=hist_len_flat, \
                dtype=tf.float32, scope='hist_layer_%d' % ii)
      # get output from last timestep
      hist_enc = support.last_relevant(hist_out, hist_len_flat)

      # reshape back
      size = [-1, hist_size[1], self.params['lstm_size']]
      hist_enc = tf.reshape(hist_enc, size)

      # concatenate, mlp and tanh
      num_r = self.params['num_rounds']
      # dot product
      attention = tf.matmul(ques_enc, hist_enc, transpose_b=True)

      # a very small large number
      u_mat = np.full((num_r, num_r), -1e10)
      suppress_mat = tf.constant(np.triu(u_mat, 1), dtype=tf.float32)
      l_mat = np.full((num_r, num_r), 1)
      mask_mat = tf.constant(np.tril(l_mat), dtype=tf.float32)
      attention = tf.nn.softmax(tf.multiply(attention, mask_mat)
                            + suppress_mat)
      self.att_history = attention
      att_hist_enc = tf.matmul(attention, hist_enc)

      # flatten out
      size = [-1, self.params['lstm_size']]
      att_hist_flat = tf.reshape(att_hist_enc, size)

      # concatenate attended history and encoder state for the last layer
      concat = tf.concat([encoder_states[-1].h, att_hist_flat], -1)
      new_state = LSTMStateTuple(encoder_states[-1].c,
                                 FC(concat, self.params['lstm_size']))

      # make it mutable
      encoder_states = list(encoder_states)
      encoder_states[-1] = new_state
      self.encoder_states = tuple(encoder_states)

      # check if wv flag is set
      if self.params['use_word_vectors']:
        # transform the encoder outputs for further attention alignments
        # encoder_outputs_flat has shape [T, N, lstm_dim]
        encoder_h_transformed = fc('encoder_h_transform',
          tf.reshape(embedded_seq, [-1, self.encoder_embed_dim]),
                            output_dim=lstm_dim)
      else:
        # transform the encoder outputs for further attention alignments
        # encoder_outputs_flat has shape [T, N, lstm_dim]
        encoder_h_transformed = fc('encoder_h_transform',
          tf.reshape(encoder_outputs, [-1, lstm_dim]), output_dim=lstm_dim)

      encoder_h_transformed = tf.reshape(encoder_h_transformed,
                       to_T([T, N, lstm_dim]))
      self.encoder_h_transformed = encoder_h_transformed

      # seq_not_finished is a shape [T, N, 1] tensor, where seq_not_finished[t, n]
      # is 1 iff sequence n is not finished at time t, and 0 otherwise
      seq_not_finished = tf.less(tf.range(T)[:, tf.newaxis, tf.newaxis],
                     input_seq_len[:, tf.newaxis])
      seq_not_finished = tf.cast(seq_not_finished, tf.float32)
      self.seq_not_finished = seq_not_finished

  def _build_decoder(self, use_gt_layout, gt_layout_batch, scope='decoder',
    reuse=None):
    # The main difference from before is that the decoders now takes another
    # input (the attention) when computing the next step
    # T_max is the maximum length of decoded sequence (including <eos>)
    #
    # This function is for decoding only. It performs greedy search or sampling.
    # the first input is <go> (its embedding vector) and the subsequent inputs
    # are the outputs from previous time step
    # num_vocab does not include <go>
    #
    # use_gt_layout is None or a bool tensor, and gt_layout_batch is a tenwor
    # with shape [T_max, N].
    # If use_gt_layout is not None, then when use_gt_layout is true, predict
    # exactly the tokens in gt_layout_batch, regardless of actual probability.
    # Otherwise, if sampling is True, sample from the token probability
    # If sampling is False, do greedy decoding (beam size 1)
    N = self.N
    encoder_states = self.encoder_states
    T_max = self.T_decoder
    lstm_dim = self.lstm_dim
    num_layers = self.num_layers
    apply_dropout = self.decoder_dropout
    EOS_token = self.EOS_token
    sampling = self.decoder_sampling

    with tf.variable_scope(scope, reuse=reuse):
      embedding_mat = tf.get_variable('embedding_mat',
        [self.decoder_num_vocab, self.decoder_embed_dim])
      # we use a separate embedding for <go>, as it is only used in the
      # beginning of the sequence
      go_embedding = tf.get_variable('go_embedding', [1, self.decoder_embed_dim])

      with tf.variable_scope('att_prediction'):
        v = tf.get_variable('v', [lstm_dim])
        W_a = tf.get_variable('weights', [lstm_dim, lstm_dim],
          initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable('biases', lstm_dim,
          initializer=tf.constant_initializer(0.))

      # The parameters to predict the next token
      with tf.variable_scope('token_prediction'):
        W_y = tf.get_variable('weights', [lstm_dim*2, self.decoder_num_vocab],
          initializer=tf.contrib.layers.xavier_initializer())
        b_y = tf.get_variable('biases', self.decoder_num_vocab,
          initializer=tf.constant_initializer(0.))

      # Attentional decoding
      # Loop function is called at time t BEFORE the cell execution at time t,
      # and its next_input is used as the input at time t (not t+1)
      # c.f. https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn
      mask_range = tf.reshape(tf.range(self.decoder_num_vocab, dtype=tf.int32),
                              [1, -1])

      if use_gt_layout is not None:
        gt_layout_mult = tf.cast(use_gt_layout, tf.int32)
        pred_layout_mult = 1 - gt_layout_mult
      def loop_fn(time, cell_output, cell_state, loop_state):
        if cell_output is None:  # time == 0
          next_cell_state = encoder_states
          next_input = tf.tile(go_embedding, to_T([N, 1]))
        else:  # time > 0
          next_cell_state = cell_state

          # compute the attention map over the input sequence
          # a_raw has shape [T, N, 1]
          att_raw = tf.reduce_sum(
            tf.tanh(tf.nn.xw_plus_b(cell_output, W_a, b_a) +
                self.encoder_h_transformed) * v,
            axis=2, keep_dims=True)
          # softmax along the first dimension (T) over not finished examples
          # att has shape [T, N, 1]
          att = tf.nn.softmax(att_raw, dim=0)*self.seq_not_finished
          att = att / tf.reduce_sum(att + 1e-10, axis=0, keep_dims=True)
          # d has shape [N, lstm_dim]
          d2 = tf.reduce_sum(att*self.encoder_outputs, axis=0)

          # token_scores has shape [N, num_vocab]
          token_scores = tf.nn.xw_plus_b(
            tf.concat([cell_output, d2], axis=1),
            W_y, b_y)

          decoding_state = loop_state[2]
          # token_validity has shape [N, num_vocab]
          token_validity = _get_valid_tokens(decoding_state, self.W, self.b)
          token_validity.set_shape([None, self.decoder_num_vocab])
          if use_gt_layout is not None:
            # when there's ground-truth layout, do not re-normalize prob
            # and treat all tokens as valid
            token_validity = tf.logical_or(token_validity, use_gt_layout)

          validity_mult = tf.cast(token_validity, tf.float32)

          # predict the next token (behavior depending on parameters)
          if sampling:
            token_scores_valid = token_scores - (1-validity_mult) * 50
            # TODO:debug
            sampled_token = tf.cast(tf.reshape(
                tf.multinomial(token_scores_valid/self.temperature, 1), [-1]), tf.int32)

            # make sure that the predictions are ALWAYS valid 
            # (it can be invalid with very small prob)
            # If not, just fall back to min cases
            # pred_mask has shape [N, num_vocab]
            sampled_mask = tf.equal(mask_range, tf.reshape(sampled_token, [-1, 1]))
            is_sampled_valid = tf.reduce_any(
              tf.logical_and(sampled_mask, token_validity),
              axis=1)

            # Fall back to max score (no sampling)
            min_score = tf.reduce_min(token_scores)
            token_scores_valid = tf.where(token_validity, token_scores,
                           tf.ones_like(token_scores)*(min_score-1))
            max_score_token = tf.cast(tf.argmax(token_scores_valid, 1), tf.int32)
            predicted_token = tf.where(is_sampled_valid, sampled_token, max_score_token)
          else:
            min_score = tf.reduce_min(token_scores)
            token_scores_valid = tf.where(token_validity, token_scores,
                           tf.ones_like(token_scores)*(min_score-1))
            # predicted_token has shape [N]
            predicted_token = tf.cast(tf.argmax(token_scores_valid, 1), tf.int32)
          if use_gt_layout is not None:
            predicted_token = (gt_layout_batch[time-1] * gt_layout_mult
                     + predicted_token * pred_layout_mult)

          # a robust version of softmax
          # all_token_probs has shape [N, num_vocab]
          all_token_probs = tf.nn.softmax(token_scores) * validity_mult
          # tf.check_numerics(all_token_probs, 'NaN/Inf before div')
          all_token_probs = all_token_probs / tf.reduce_sum(all_token_probs + 1e-10, axis=1, keep_dims=True)
          # tf.check_numerics(all_token_probs, 'NaN/Inf after div')

          # mask has shape [N, num_vocab]
          mask = tf.equal(mask_range, tf.reshape(predicted_token, [-1, 1]))
          # token_prob has shape [N], the probability of the predicted token
          # although token_prob is not needed for predicting the next token
          # it is needed in output (for policy gradient training)
          # [N, num_vocab]
          token_prob = tf.reduce_sum(all_token_probs * tf.cast(mask, tf.float32), axis=1)
          # tf.assert_positive(token_prob)
          neg_entropy = tf.reduce_sum(
            all_token_probs * tf.log(all_token_probs + (1-validity_mult) + 1e-10),
            axis=1)

          # update states
          updated_decoding_state = _update_decoding_state(
            decoding_state, predicted_token, self.P)

          # the prediction is from the cell output of the last step
          # timestep (t-1), feed it as input into timestep t
          next_input = tf.nn.embedding_lookup(embedding_mat, predicted_token)

        elements_finished = tf.greater_equal(time, T_max)

        # loop_state is a 5-tuple, representing
        #   1) the predicted_tokens
        #   2) the prob of predicted_tokens
        #   3) the decoding state (used for validity)
        #   4) the negative entropy of policy (accumulated across timesteps)
        #   5) the attention
        if loop_state is None:  # time == 0
          # Write the predicted token into the output
          predicted_token_array = tf.TensorArray(dtype=tf.int32, size=T_max,
            infer_shape=False)
          token_prob_array = tf.TensorArray(dtype=tf.float32, size=T_max,
            infer_shape=False)
          init_decoding_state = tf.tile(to_T([[0, 0, T_max]], dtype=tf.int32), to_T([N, 1]))
          att_array = tf.TensorArray(dtype=tf.float32, size=T_max,
            infer_shape=False)
          next_loop_state = (predicted_token_array,
                   token_prob_array,
                   init_decoding_state,
                   tf.zeros(to_T([N]), dtype=tf.float32),
                   att_array)
        else:  # time > 0
          t_write = time-1
          next_loop_state = (loop_state[0].write(t_write, predicted_token),
                   loop_state[1].write(t_write, token_prob),
                   updated_decoding_state,
                   loop_state[3] + neg_entropy,
                   loop_state[4].write(t_write, att))
        return (elements_finished, next_input, next_cell_state, cell_output,
            next_loop_state)

      # The RNN
      cell = _get_lstm_cell(num_layers, lstm_dim, apply_dropout)
      _, _, decodes_ta = tf.nn.raw_rnn(cell, loop_fn, scope='lstm')
      predicted_tokens = decodes_ta[0].stack()
      token_probs = decodes_ta[1].stack()
      neg_entropy = decodes_ta[3]
      # atts has shape [T_decoder, T_encoder, N, 1]
      atts = decodes_ta[4].stack()
      # static dimension recast
      atts = tf.reshape(atts, [self.T_decoder, self.T_encoder, -1, 1])
      self.atts = atts
      # word_vec has shape [T_decoder, N, 1]
      word_vecs = tf.reduce_sum(atts*self.embedded_input_seq, axis=1)

      predicted_tokens.set_shape([None, None])
      token_probs.set_shape([None, None])
      neg_entropy.set_shape([None])
      #word_vecs.set_shape([None, None, self.encoder_embed_dim])
      # static shapes
      word_vecs.set_shape([self.T_decoder, None, self.encoder_embed_dim])

      self.predicted_tokens = predicted_tokens
      self.token_probs = token_probs
      self.neg_entropy = neg_entropy
      self.word_vecs = word_vecs
#------------------------------------------------------------------------------
