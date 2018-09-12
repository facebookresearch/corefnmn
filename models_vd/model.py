"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Main CorefNMN model class.

Explicit visual coreference resolution in visual dialog using neural module
networks. Takes parameters and assemblers as input.

Author: Satwik Kottur
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_fold as td

from models_vd.generator import ProgramGenerator
from models_vd.executor import ProgramExecutor
from models_vd.decoder import AnswerDecoder
from util import support


class CorefNMN:
  def __init__(self, params, assemblers, reuse=None):
    # train mode
    params['train_mode'] = 'test_split' not in params
    print('Building model with train_model as: ' + str(params['train_mode']))

    self.params = params
    self.assemblers = assemblers
    #----------------------------------------------------------------------

    # module phases
    self.phases = ['generate_program', 'execute_program', 'generate_answer']

    # initializing input and output placeholders
    self.inputs = {ii: {} for ii in self.phases}
    self.outputs = self.inputs.copy()

    # build place holders for inputs and outputs in the tensorflow graph
    holders = self._build_placeholders(params)
    self.holders = holders

    with tf.variable_scope(params['model'], reuse=reuse):
      # keep track of all outputs
      output_pool = {}

      # Part 1: Seq2seq RNN to generate module layout tokens
      with tf.variable_scope('generate_program'):
        self.generator = ProgramGenerator(holders, assemblers['ques'], params)
        self.inputs['generate_program'] = self.generator.get_inputs()
        self.outputs['generate_program'] = self.generator.get_outputs()
        # add outputs to pool
        output_pool.update(self.outputs['generate_program'])

      # Part 2: Neural Module Network
      with tf.variable_scope('execute_program'):
        self.executor = ProgramExecutor(holders, output_pool,
                                        assemblers['cap'], params)
        self.inputs['execute_program'] = self.executor.get_inputs()
        self.outputs['execute_program'] = self.executor.get_outputs()
        # add outputs to pool
        output_pool.update(self.outputs['execute_program'])

      # Part 3: Seq2Seq decoding of the answer
      with tf.variable_scope('generate_answer'):
        self.decoder = AnswerDecoder(holders, output_pool, params)

        self.inputs['generate_answer'] = self.decoder.get_inputs()
        self.outputs['generate_answer'] = self.decoder.get_outputs()

    # pool up all the outputs
    pooled_dict = []
    outputs = self.outputs.copy()
    for ii in outputs:
      pooled_dict += outputs[ii].items()
    self.pooled_outputs = dict(pooled_dict)

  #---------------------------------------------------------------------------
  def _build_placeholders(self, params):
    inputs = {}

    # Phase 1 - program generation
    size = [params['max_enc_len'], None]
    inputs['ques'] = tf.placeholder(tf.int32, size, 'ques')
    inputs['ques_len'] = tf.placeholder(tf.int32, [None], 'ques_len')
    inputs['prog_gt'] = tf.placeholder(tf.int32, [None, None], 'prog')

    size = [None, params['max_enc_len']]
    inputs['cap'] = tf.placeholder(tf.int32, size, 'caption')
    inputs['cap_len'] = tf.placeholder(tf.int32, [None], 'cap_len')
    inputs['cap_prog_gt'] = tf.placeholder(tf.int32, [None, None],
                                           'cap_prog_gt')

    # mask for pairwise program token loss
    inputs['prog_att_mask'] = tf.placeholder(tf.float32, [None, None, None],
                                             'mask')
    # for supervising placeholders
    if params['supervise_attention']:
      size = [params['max_dec_len'], params['max_enc_len'], None, 1]
      inputs['prog_att_gt'] = tf.placeholder(tf.float32, size, 'gt_att')
      inputs['cap_att_gt'] = tf.placeholder(tf.float32, size, 'cap_att')
      # masking out relevant parts for complete supervision
      inputs['ques_super_mask'] = tf.placeholder(tf.float32, size, 'q_mask')
      inputs['cap_super_mask'] = tf.placeholder(tf.float32, size, 'c_mask')
      inputs['supervise_switch'] = tf.placeholder(tf.bool, 'supervise_switch')

    # tie encoder and decoder
    size = [params['num_layers'], None, params['lstm_size']]
    inputs['enc_dec_h'] = tf.placeholder(tf.float32, size, 'enc_dec_h')
    inputs['enc_dec_c'] = tf.placeholder(tf.float32, size, 'enc_dec_c')

    # Phase 2 - program execution
    size = [None, params['h_feat'], params['w_feat'], params['d_feat']]
    inputs['img_feat'] = tf.placeholder(tf.float32, size, 'img_feat')
    inputs['prog_validity'] = tf.placeholder(tf.bool, [None])

    # Phase 2.5 - caption execution
    inputs['align_gt'] = tf.placeholder(tf.int32, [None], 'align_cap')
    inputs['prog_validity_cap'] = tf.placeholder(tf.bool, [None])

    # Phase 3 - answer generation
    inputs['ans_in'] = tf.placeholder(tf.int32, [None, None], 'ans_in')
    inputs['ans_out'] = tf.placeholder(tf.int32, [None, None], 'ans_out')
    inputs['ans'] = tf.placeholder(tf.int32, [None, None], 'ans')
    inputs['ans_len'] = tf.placeholder(tf.int32, [None], 'ans_len')

    # if discriminative, encode options
    # NOTE: num_options hard coded to 100
    num_options = 100
    size = [None, params['max_enc_len'], num_options]
    inputs['opt'] = tf.placeholder(tf.int32, size, 'opt_out')
    inputs['opt_len'] = tf.placeholder(tf.int32, [None, num_options], 'opt_len')
    inputs['gt_ind'] = tf.placeholder(tf.int32, [None], 'gt_ind')

    # history
    size = [None, params['num_rounds'], 2 * params['max_enc_len']]
    inputs['hist'] = tf.placeholder(tf.int32, size, 'history')
    size = [None, params['num_rounds']]
    inputs['hist_len'] = tf.placeholder(tf.int32, size, 'hist_len')

    # place holders for fact
    size = [None, params['max_enc_len']]
    inputs['fact'] = tf.placeholder(tf.int32, size, 'fact')
    inputs['fact_len'] = tf.placeholder(tf.int32, [None], 'fact_len')

    if not self.params['train_mode']:
      # additional placeholders during evaluation
      size = [None, params['lstm_size']]
      inputs['context'] = tf.placeholder(tf.float32, size, 'context')
      size = [1, 1, None, params['lstm_size']]
      inputs['cap_enc'] = tf.placeholder(tf.float32, size, 'cap_enc')
      size = [None, None, None, params['lstm_size']]
      inputs['ques_enc'] = tf.placeholder(tf.float32, size, 'ques_enc')
      size = [None, params['lstm_size']]
      inputs['hist_enc'] = tf.placeholder(tf.float32, size, 'hist_enc')
      size = [params['max_dec_len'], None, params['text_embed_size']]
      inputs['ques_attended'] = tf.placeholder(tf.float32, size, 'ques_att')
      inputs['cap_attended'] = tf.placeholder(tf.float32, size, 'cap_att')

    return inputs
  #---------------------------------------------------------------------------

  # method to initialize training related attributes
  def setup_training(self):
    # answer prediction loss
    total_loss = self.outputs['generate_answer']['ans_token_loss']

    # supervised sequence prediction loss
    total_loss += self.outputs['generate_program']['prog_pred_loss']

    if 'nmn-cap' in self.params['model'] and self.params['cap_alignment']:
      total_loss += self.outputs['execute_program']['cap_align_loss']

    # add the total loss to the list of outputs
    self.pooled_outputs['total_loss'] = total_loss

  # setters and getters
  def get_total_loss(self):
    return self.pooled_outputs['total_loss']

  def add_solver_op(self, op):
    self.pooled_outputs['solver'] = op
  #---------------------------------------------------------------------------

  def run_train_iteration(self, batch, sess):
    iter_loss = {}

    # collect feeds from all subcomponents
    feeder = self.generator.produce_feed_dict(batch)
    feeder.update(self.executor.produce_feed_dict(batch))
    feeder.update(self.decoder.produce_feed_dict(batch))

    # run all subcomponents together
    output = sess.run(self.pooled_outputs, feed_dict=feeder)

    # record all the loss values
    iter_loss['prog'] = output['prog_pred_loss']
    if 'nmn-cap' in self.params['model']:
      iter_loss['align'] = output['cap_align_loss']
    else:
      iter_loss['align'] = 0.
    iter_loss['ans'] = output['ans_token_loss']
    iter_loss['total'] = output['total_loss']

    return iter_loss, None
  #---------------------------------------------------------------------------

  def run_evaluate_iteration(self, batch, sess, eval_options=True):
    # Part 0 & 1: Run Convnet and generate module layout
    feeder = self.generator.produce_feed_dict(batch)
    output = sess.run(self.outputs['generate_program'], feed_dict=feeder)

    # Part 2: Run NMN and learning steps
    feeder = self.executor.produce_feed_dict(batch, output)
    output.update(sess.run(self.outputs['execute_program'], feed_dict=feeder))

    if 'pred_tokens' in output:
      output['matches'] = [batch['gt_layout'] == output['pred_tokens']]

    # if options are not to be scored
    if not eval_options: return None, outputs

    # Part 3: Run the answer generation language model (disc | gen)
    if self.params['decoder'] == 'gen':
      option_batch = output.copy()
      option_batch.update(batch)
      phase_output = self.outputs['generate_answer']['llh']

      num_options = len(batch['opt_len'])
      batch_size = batch['opt_len'][0].shape[0]
      option_scores = np.zeros((batch_size, num_options))

      option_probs = np.zeros((batch_size, num_options))
      for opt_id in range(num_options):
        option_batch['ans_in'] = batch['opt_in'][opt_id]
        option_batch['ans_out'] = batch['opt_out'][opt_id]
        option_batch['ans_len'] = batch['opt_len'][opt_id]

        feeder = self.decoder.produce_feed_dict(option_batch, output)
        scores = sess.run(phase_output, feed_dict=feeder)
        option_scores[:, opt_id] = scores

    # Part 3: Run the decoder model
    elif self.params['decoder'] == 'disc':
      batch_size = batch['opt_len'][0].shape[0]
      feeder = self.decoder.produce_feed_dict(batch, output)
      output.update(sess.run(self.outputs['generate_answer'], feeder))
      option_scores = output['scores']

    # extract ground truth score, and get ranks
    gt_scores = option_scores[(range(batch_size), batch['gt_ind'])]
    ranks = np.sum(option_scores > gt_scores.reshape(-1, 1), axis=1) + 1

    output['scores'] = option_scores

    return ranks, output
  #---------------------------------------------------------------------------

  def run_visualize_iteration(self, batch, sess, eval_options=True):
    output = batch.copy()

    # Part 0 & 1: Run Convnet and generate module layout
    feeder = self.generator.produce_feed_dict(batch)
    output.update(sess.run(self.outputs['generate_program'], feeder))

    # Part 2: Run NMN and learning steps
    feeder = self.executor.produce_feed_dict(batch, output, True)
    output.update(sess.run(self.outputs['execute_program'], feeder))

    # segregate weights and attention maps 
    output['intermediates'] = self.executor.segregrate_outputs(output)

    if not eval_options: return None, output

    # Part 3: Run the answer generation language model
    if self.params['decoder'] == 'gen':
      option_batch = output.copy()
      option_batch.update(batch)
      phase_output = self.outputs['generate_answer']['llh']

      # Part 3: Run the answer generation language model for each option
      num_options = len(batch['opt_len'])
      batch_size = batch['opt_len'][0].shape[0]
      option_scores = np.zeros((batch_size, num_options))
      for opt_id in range(num_options):
        option_batch['ans_in'] = batch['opt_in'][opt_id]
        option_batch['ans_out'] = batch['opt_out'][opt_id]
        option_batch['ans_len'] = batch['opt_len'][opt_id]

        feeder = self.decoder.produce_feed_dict(option_batch, output)
        scores = sess.run(phase_output, feed_dict=feeder)
        option_scores[:, opt_id] = scores

    # Part 3: Run the decoder model
    elif self.params['decoder'] == 'disc':
      batch_size = batch['opt_len'][0].shape[0]
      feeder = self.decoder.produce_feed_dict(batch, output)
      output.update(sess.run(self.outputs['generate_answer'], feeder))
      option_scores = output['scores']

    # extract ground truth score, and get ranks
    gt_scores = option_scores[(range(batch_size), batch['gt_ind'])]
    ranks = np.sum(option_scores > gt_scores.reshape(-1, 1), axis=1) + 1

    output['scores'] = option_scores
    return ranks, output
#-------------------------------------------------------------------------
