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

from models_mnist.generator import ProgramGenerator
from models_mnist.executor import ProgramExecutor
from models_mnist.decoder import AnswerDecoder
from util import support


class CorefNMN:
  def __init__(self, params, assemblers, reuse=None):
    # train mode
    params['train_mode'] = 'test_split' not in params
    print('Building model with train_model as: ' + str(params['train_mode']))

    self.params = params
    self.assemblers = assemblers

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
                                        assemblers['ques'], params)
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

    # place holders for fact
    size = [None, params['max_enc_len'] + 1]
    inputs['fact'] = tf.placeholder(tf.int32, size, 'fact')
    inputs['fact_len'] = tf.placeholder(tf.int32, [None], 'fact_len')

    # tie encoder and decoder
    size = [params['num_layers'], None, params['lstm_size']]
    inputs['enc_dec_h'] = tf.placeholder(tf.float32, size, 'enc_dec_h')
    inputs['enc_dec_c'] = tf.placeholder(tf.float32, size, 'enc_dec_c')

    # Phase 2 - program execution
    size = [None, 112, 112, 3]
    inputs['image'] = tf.placeholder(tf.float32, size, 'image')
    inputs['prog_validity'] = tf.placeholder(tf.bool, [None])

    # for the answer indices
    inputs['ans_ind'] = tf.placeholder(tf.int32, [None], 'ans_ind')

    # history
    size = [None, params['num_rounds'], params['max_enc_len'] + 1]
    inputs['hist'] = tf.placeholder(tf.int32, size, 'history')
    size = [None, params['num_rounds']]
    inputs['hist_len'] = tf.placeholder(tf.int32, size, 'hist_len')

    if not self.params['train_mode']:
      # additional placeholders during evaluation
      size = [None, params['lstm_size']]
      inputs['context'] = tf.placeholder(tf.float32, size, 'context')
      size = [None, None, None, params['lstm_size']]
      inputs['ques_enc'] = tf.placeholder(tf.float32, size, 'ques_enc')
      size = [None, params['lstm_size']]
      inputs['hist_enc'] = tf.placeholder(tf.float32, size, 'hist_enc')
      size = [params['max_dec_len'], None, params['text_embed_size']]
      inputs['ques_attended'] = tf.placeholder(tf.float32, size, 'ques_att')

    return inputs
  #---------------------------------------------------------------------------

  # method to initialize training related attributes
  def setup_training(self):
    # answer prediction loss
    total_loss = self.outputs['generate_answer']['ans_token_loss']

    # supervised sequence prediction loss
    total_loss += self.outputs['generate_program']['prog_pred_loss']

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
      prog_matches = []
      prog_matches.append(batch['gt_layout'] == output['pred_tokens'])
      output['matches'] = prog_matches

    # Part 3: Run the answer generation language model
    feeder = self.decoder.produce_feed_dict(batch, output)
    output.update(sess.run(self.outputs['generate_answer'], feeder))

    # use the logits and get the prediction
    matches = np.argmax(output['logits'], 1) == batch['ans_ind']

    return matches, output
  #---------------------------------------------------------------------------

  def run_visualize_iteration(self, batch, sess, eval_options=True):
    output = batch.copy()

    # Part 0 & 1: Run Convnet and generate module layout
    feeder = self.generator.produce_feed_dict(batch)
    output.update(sess.run(self.outputs['generate_program'], feeder))

    # Part 2: Run NMN and learning steps
    feeder = self.executor.produce_feed_dict(batch, output, True)
    output.update(sess.run(self.outputs['execute_program'], feeder))

    # Part 3: Run the answer generation language model
    feeder = self.decoder.produce_feed_dict(batch, output)
    output.update(sess.run(self.outputs['generate_answer'], feeder))

    # segregate weights and attention maps 
    output['intermediates'] = self.executor.segregrate_outputs(output)

    return None, output
#-------------------------------------------------------------------------
