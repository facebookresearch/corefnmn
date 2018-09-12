"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Main class to decode and produce an answer.

Answer decoder for explicit visual coreference resolution model in visual
dialog using neural module networks, called CorefNMN.
Support two kinds of decoders:
(a) Generative: A recurrent neural network based language model that can
    generate novel answers. At test time, all candidate answers are scored
    based on loglikelihood of the language model.

(b) Discriminative: A discriminative classifier to identify the correct
    answer from a pool of candidate options at train time.
    At test time, options are ranked based on class probabilities.

Author: Satwik Kottur
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import dropout
from tensorflow.contrib.layers import fully_connected as FC
from util import support

class AnswerDecoder:
  def __init__(self, inputs, output_pool, params):
    """Initialize answer decoder.

    Args:
      inputs:
      output_pool:
      params:
    """
    self.params = params

    # keep track of inputs and outputs
    used_inputs = []
    outputs = {}

    # alias for criterion
    criterion = tf.nn.sparse_softmax_cross_entropy_with_logits

    # begin decoding
    with tf.variable_scope(self.params['embed_scope'], reuse=True):
      size = [params['text_vocab_size'], params['text_embed_size']]
      embed_mat = tf.get_variable('embed_mat')

    output = tf.nn.embedding_lookup(embed_mat, inputs['ans_in'])
    used_inputs.extend(['ans_in', 'ans_out', 'ans_len'])

    # recurrent neural network cell
    cell = tf.contrib.rnn.BasicLSTMCell(params['lstm_size'])

    # decide the source based on train / evaluation
    source = output_pool if params['train_mode'] else inputs

    # concatenate question to both
    concat_list = []
    # add program context vector
    concat_list.append(source['context'])
    # adding last hidden size
    concat_list.append(source['enc_dec_h'][-1])
    used_inputs.extend(['enc_dec_h', 'enc_dec_c'])

    if not params['train_mode']:
      used_inputs.append('context')
    #--------------------------------------------------------------------------

    # stack all the vectors
    stack_vec = tf.concat(concat_list, axis=1)
    stack_vec = FC(stack_vec, params['lstm_size'])

    # construct encoder decoder H
    enc_dec_h = [source['enc_dec_h'][ii]
                 for ii in range(params['num_layers'] - 1)]
    enc_dec_h.append(stack_vec)
    # construct encoder decoder C
    enc_dec_c = [source['enc_dec_c'][ii] for ii in range(params['num_layers'])]
    init_state = [tf.contrib.rnn.LSTMStateTuple(cc, hh)
                  for cc, hh in zip(enc_dec_c, enc_dec_h)]

    if params['decoder'] == 'gen':
      for ii in range(params['num_layers']):
        # dynamic rnn
        output,  _ = tf.nn.dynamic_rnn(cell, output,
                                       sequence_length=inputs['ans_len'],
                                       initial_state=init_state[ii],
                                       dtype=tf.float32, scope='layer_%d' % ii)

      # predict the output words
      output = FC(output, params['text_vocab_size'], activation_fn=None)
      # create a mask
      mask = tf.not_equal(inputs['ans_out'], params['pad_id'])
      mask = tf.cast(mask, tf.float32)

      # multiply by mask for variable length sequences
      answer_loss = criterion(logits=output, labels=inputs['ans_out'])
      masked_answer_loss = tf.multiply(answer_loss, mask)
      token_likelihood = tf.reduce_sum(masked_answer_loss)
      num_tokens = tf.maximum(tf.reduce_sum(mask), 1)

      outputs['ans_token_loss'] = token_likelihood/num_tokens
      outputs['per_sample_loss'] = tf.reduce_sum(masked_answer_loss, 1)

      # extract the probabilities
      out_softmax = tf.nn.log_softmax(output)
      out_softmax_flat = tf.reshape(out_softmax, [-1, params['text_vocab_size']])
      orig_shape = tf.shape(inputs['ans_out'])
      ans_out_flat = tf.reshape(inputs['ans_out'], [-1])
      inds = [tf.range(0, tf.shape(ans_out_flat)[0]), ans_out_flat]
      inds = tf.stack(inds, axis=1)

      prob_tokens = tf.gather_nd(out_softmax_flat, inds)
      prob_tokens = tf.reshape(prob_tokens, orig_shape)
      prob_tokens = tf.multiply(prob_tokens, mask)
      # compute the loglikelihood
      outputs['llh'] = tf.reduce_sum(prob_tokens, 1)
      # compute mean instead of sum
      num_tokens = tf.maximum(tf.reduce_sum(mask, 1), 1)
      outputs['llh_mean'] = outputs['llh'] / num_tokens

    elif params['decoder'] == 'disc':
      # embed options and encode via lstm
      with tf.variable_scope(self.params['embed_scope'], reuse=True):
        size = [params['text_vocab_size'], params['text_embed_size']]
        embed_mat = tf.get_variable('embed_mat')
      opt_embed = tf.nn.embedding_lookup(embed_mat, inputs['opt'])

      # transpose and merging batch and option dimension
      opt_embed = tf.transpose(opt_embed, [0, 2, 1, 3])
      shape = opt_embed.shape.as_list()
      opt_embed = tf.reshape(opt_embed, [-1, shape[2], shape[3]])

      opt_len = tf.reshape(inputs['opt_len'], [-1])

      output, _ = tf.nn.dynamic_rnn(cell, opt_embed,
                                    sequence_length=opt_len,
                                    dtype=tf.float32, scope='opt_layer_0')
      for ii in range(1, params['num_layers']):
        # dynamic rnn
        output, _ = tf.nn.dynamic_rnn(cell, output, \
                                      sequence_length=opt_len,
                                      dtype=tf.float32,
                                      scope='opt_layer_%d' % ii)

      opt_encode = support.last_relevant(output, opt_len)
      # reshape back
      opt_encode = tf.reshape(opt_encode, [-1, shape[1], params['lstm_size']])

      # score the options with context vector
      score_vec = tf.matmul(opt_encode, tf.expand_dims(stack_vec, -1))
      score_vec = tf.squeeze(score_vec, -1)
      scores = criterion(logits=score_vec, labels=inputs['gt_ind'])
      outputs['ans_token_loss'] = tf.reduce_mean(scores)
      outputs['scores'] = score_vec

      used_inputs.extend(['opt', 'opt_len', 'gt_ind'])

    # setup the inputs and outputs
    self.outputs = outputs
    self.inputs = {ii: inputs[ii] for ii in used_inputs}
  #----------------------------------------------------------------------------

  # setters and getters
  def get_outputs(self):
    return self.outputs

  def get_inputs(self):
    return self.inputs
  #----------------------------------------------------------------------------

  # produce feed dict
  def produce_feed_dict(self, batch, output_pool=None):
    """Produces the feed dict for this subcomponent.

    Args:
      batch: Batch returned from dataloader
      output_pool: Outputs from previous subcomponents, mostly when evaluating

    Returns:
      feed_dict: Returns the feed dictionary
    """

    feed_dict = {}
    for key in ['ans_in', 'ans_out', 'ans_len']:
      feed_dict[self.inputs[key]] = batch[key]

    # if not in train mode, use output_pool
    if not self.params['train_mode']:
      for key in ['context', 'enc_dec_h', 'enc_dec_c']:
        feed_dict[self.inputs[key]] = output_pool[key]

    # additional feeds for discriminative decoder
    if self.params['decoder'] == 'disc':
      feed_dict[self.inputs['opt']] = np.stack(batch['opt_out'], -1)
      feed_dict[self.inputs['opt_len']] = np.stack(batch['opt_len'], -1)
      feed_dict[self.inputs['gt_ind']] = batch['gt_ind']

    return feed_dict
  #----------------------------------------------------------------------------
