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

    # decide the source based on train / evaluation
    source = output_pool if params['train_mode'] else inputs

    # concatenate question to both
    concat_list = []
    # add program context vector
    concat_list.append(source['context'])

    # stack all the vectors
    stack_vec = tf.concat(concat_list, axis=1)

    # a linear to number of choices
    logits = FC(stack_vec, params['num_choices'], activation_fn=None)
    outputs['logits'] = logits

    # softmax over the choices
    answer_loss = criterion(logits=logits, labels=inputs['ans_ind'])
    used_inputs.append('ans_ind')
    outputs['ans_token_loss'] = tf.reduce_mean(answer_loss)

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
    feed_dict[self.inputs['ans_ind']] = batch['ans_ind']

    # if not training, use previous outputs, else inputs
    if not self.params['train_mode']:
      feeds = ['ques_enc', 'context', 'cap_enc', 'hist_enc']
      feed_dict.update({self.inputs[feed]: output_pool[feed]
                        for feed in feeds if feed in self.inputs})

    return feed_dict
