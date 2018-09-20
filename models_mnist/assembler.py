"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

TODO(satwik): Write a description about what this file contains and what
it does.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from tensorflow import convert_to_tensor as to_T
from models_mnist import modules as lm

# the number of attention input to each module
_module_input_num = {
                     '_Find': 0,
                     '_Refer': 0,
                     '_Exclude': 0,
                     '_Transform': 1,
                     '_Exist': 1,
                     '_Count': 1,
                     '_And': 2,
                     '_Diff': 2,
                     '_Not': 1,
                     '_Describe': 1
                    }

# output type of each module
_module_output_type = {
                       '_Find': 'att',
                       '_Refer': 'att',
                       '_Exclude': 'att',
                       '_Exist': 'ans',
                       '_Count': 'ans',
                       '_Transform': 'att',
                       '_And': 'att',
                       '_Diff': 'att',
                       '_Not': 'att',
                       '_Describe': 'ans'
                      }

INVALID_EXPR = 'INVALID_EXPR'

# decoding validity: maintaining a state x of [#att, #ans, T_remain]
# when T_remain is T_decoder when decoding the first module token
# a token s can be predicted iff all(<x, w_s> - b_s >= 0)
# the validity token list is
#       XW - b >= 0
# the state transition matrix is P, so the state update is X += S P,
# where S is the predicted tokens (one-hot vectors)

def _build_validity_mats(module_names):
  state_size = 3
  num_vocab_nmn = len(module_names)
  num_constraints = 4
  P = np.zeros((num_vocab_nmn, state_size), np.int32)
  W = np.zeros((state_size, num_vocab_nmn, num_constraints), np.int32)
  b = np.zeros((num_vocab_nmn, num_constraints), np.int32)

  # collect the input and output numbers of each module
  att_in_nums = np.zeros(num_vocab_nmn)
  att_out_nums = np.zeros(num_vocab_nmn)
  ans_out_nums = np.zeros(num_vocab_nmn)
  for n_s, s in enumerate(module_names):
    if s != '<eos>':
      att_in_nums[n_s] = _module_input_num[s]
      att_out_nums[n_s] = _module_output_type[s] == 'att'
      ans_out_nums[n_s] = _module_output_type[s] == 'ans'
  # construct the trasition matrix P
  for n_s, s in enumerate(module_names):
    P[n_s, 0] = att_out_nums[n_s] - att_in_nums[n_s]
    P[n_s, 1] = ans_out_nums[n_s]
    P[n_s, 2] = -1
  # construct the validity W and b
  att_absorb_nums = (att_in_nums - att_out_nums)
  max_att_absorb_nonans = np.max(att_absorb_nums * (ans_out_nums == 0))
  max_att_absorb_ans = np.max(att_absorb_nums * (ans_out_nums != 0))
  for n_s, s in enumerate(module_names):
    if s != '<eos>':
      # constraint: a non-<eos> module can be outputted iff all the following 
      # hold:
      # * 0) there's enough att in the stack
      #      #att >= att_in_nums[n_s]
      W[0, n_s, 0] = 1
      b[n_s, 0] = att_in_nums[n_s]

      # * 1) for answer modules, there's no extra att in the stack
      #      #att <= att_in_nums[n_s]
      #      -#att >= -att_in_nums[n_s]
      #      for non-answer modules, T_remain >= 3
      #      (the last two has to be AnswerType and <eos>)
      if ans_out_nums[n_s] != 0:
        W[0, n_s, 1] = -1
        b[n_s, 1] = -att_in_nums[n_s]
      else:
        W[2, n_s, 1] = 1
        b[n_s, 1] = 3

      # * 2) there's no answer in the stack (otherwise <eos> only)
      #      #ans <= 0
      #      -#ans >= 0
      W[1, n_s, 2] = -1

      # * 3) there's enough time to consume the all attentions, output answer 
      #      plus <eos>
      #      3.1) for non-answer modules, we already have T_remain>= 3 from 
      #           constraint 2
      #           In maximum (T_remain-3) further steps
      #           (plus 3 steps for this, ans, <eos>) to consume atts
      #           (T_remain-3) * max_att_absorb_nonans + max_att_absorb_ans + 
      #            att_absorb_nums[n_s] >= #att
      #           T_remain*MANA - #att >= 3*MANA - MAA - A[s]
      #           - #att + MANA * T_remain >= 3*MANA - MAA - A[s]
      #      3.2) for answer modules, if it can be decoded then constraint 0&1 
      #           ensures that there'll be no att left in stack after decoding 
      #           this answer, hence no further constraints here
      if ans_out_nums[n_s] == 0:
        W[0, n_s, 3] = -1
        W[2, n_s, 3] = max_att_absorb_nonans
        b[n_s, 3] = (3 * max_att_absorb_nonans - max_att_absorb_ans -
                    att_absorb_nums[n_s])

    else:  # <eos>-case
      # constraint: a <eos> token can be outputted iff all the following holds
      # * 0) there's ans in the stack
      #      #ans >= 1
      W[1, n_s, 0] = 1
      b[n_s, 0] = 1

  return P, W, b
#------------------------------------------------------------------------------

class Assembler:
  def __init__(self, module_vocab_file):
    # read the module list, and record the index of each module and <eos>
    with open(module_vocab_file) as f:
      self.module_names = [s.strip() for s in f.readlines()]
    # find the index of <eos>
    for n_s in range(len(self.module_names)):
      if self.module_names[n_s] == '<eos>':
        self.EOS_idx = n_s
        break
    # build a dictionary from module name to token index
    self.name2idx_dict = {name: n_s for n_s, name in enumerate(self.module_names)}
    self.num_vocab_nmn = len(self.module_names)

    self.P, self.W, self.b = _build_validity_mats(self.module_names)

  def module_list2tokens(self, module_list, T=None):
    layout_tokens = [self.name2idx_dict[name] for name in module_list]
    if T is not None:
      if len(module_list) >= T:
        raise ValueError('Not enough time steps to add <eos>')
      layout_tokens += [self.EOS_idx]*(T-len(module_list))
    return layout_tokens

  def _layout_tokens2str(self, layout_tokens):
    return ' '.join([self.module_names[idx] for idx in layout_tokens])

  def assemble_refer(self, text_att, round_id, reuse_stack):
    # aliases
    weaver = self.weaver
    executor = self.executor

    # compute the scores
    logits = []
    for find_arg in reuse_stack:
      # compute the weights for each of the attention map
      inputs = (text_att, find_arg[1], round_id, find_arg[2])
      logits.append(weaver.align_text(*inputs))

    # exponential each logit
    weights = []
    for ii in logits: weights.append(weaver.exp(ii))

    # normalize the weights
    if len(weights) < 2:
      norm = weights[0]
    else:
      norm = weaver.add(weights[0], weights[1])
      for ii in weights[2:]: norm = weaver.add(norm, ii)
    for index, ii in enumerate(weights):
      weights[index] = weaver.divide(ii, norm)

    # multiply the attention with softmax weight
    prev_att = []
    for (att, _, _, _, _), weight in zip(reuse_stack, weights):
      prev_att.append(weaver.weight_attention(att, weight))

    # add all attentions to get the result
    if len(prev_att) < 2: out = prev_att[0]
    else:
      out = weaver.add_attention(prev_att[0], prev_att[1])
      for ii in prev_att[2:]:
        out = weaver.add_attention(out, ii)

    return out, weights, logits

  def assemble_exclude(self, text_att, round_id, reuse_stack):
    # aliases
    weaver = self.weaver
    executor = self.executor

    # compute the scores
    weights = []
    exclude_att = reuse_stack[0][0]
    if len(reuse_stack) > 1:
      for find_arg in reuse_stack:
        exclude_att = weaver.max_attention(exclude_att, find_arg[0])

    return weaver.normalize_exclude(exclude_att)

  # code to check if the program makes sense
  # typically contains all the checks from the _assemble_program method
  def sanity_check_program(self, layout):
    decode_stack = []
    for t_id, cur_op_id in enumerate(layout):
      cur_op_name = self.module_names[cur_op_id]
      # <eos> would mean stop
      if cur_op_id == self.EOS_idx: break

      # insufficient number of inputs
      num_inputs = _module_input_num[cur_op_name]
      if len(decode_stack) < num_inputs:
        return False, 'Insufficient inputs'

      # read the inputs
      inputs = []
      for ii in range(num_inputs):
        arg_type = decode_stack.pop()
        # cannot consume anything but attention
        if arg_type != 'att':
          return False, 'Intermediate not attention'

      decode_stack.append(_module_output_type[cur_op_name])

    # Check if only one element is left
    if len(decode_stack) != 1:
      return False, 'Left with more than one outputs'
    # final output is not answer type
    elif decode_stack[0] != 'ans':
      return False, 'Final output not an answer'

    return True, 'Valid program'

  def assemble(self, layout_tokens, executor, visualize=False):
    # layout_tokens_batch is a numpy array with shape [T, N],
    # containing module tokens and <eos>, in Reverse Polish Notation.

    # internalize executor and weaver
    self.executor = executor
    # build a weaver
    weaver = executor.create_weaver()
    self.weaver = weaver
    # visualize flag
    self.visualize = visualize

    # get extent of layout tokens
    max_time, batch_size = layout_tokens['ques'].shape
    num_rounds = executor.params['num_rounds']
    batch_size = batch_size // num_rounds
    outputs = []
    reuse = [None] * batch_size
    ques_invalid_prog = []

    # program on questions and captions, if needed
    ques_tokens = layout_tokens['ques']
    for b_id in range(batch_size):
      image = weaver.batch_input(executor._loom_types['image'], b_id)
      if executor.params['use_fact']:
        fact = weaver.batch_input(executor._loom_types['fact'], b_id)
      else: fact = None

      # Now run program on questions
      text = weaver.batch_input(executor._loom_types['text'], b_id)
      text_feat = weaver.batch_input(executor._loom_types['text_feat'], b_id)

      # collect root node outputs for down the rounds
      # tuples are immutable, recreate to ensure caption is round 0
      round_zero = weaver.batch_input(executor._loom_types['round'], 0)
      tokens = ques_tokens[:, num_rounds*b_id : num_rounds*(b_id+1)]
      inputs = (image, text, fact, text_feat, tokens, [])
      out, _, invalid_prog = self._assemble_program(*inputs)
      ques_invalid_prog.extend(invalid_prog)

      outputs.extend(out['comp'])
      if visualize:
        outputs.extend([ii for ii, _ in out['vis']['att']])
        outputs.extend(out['vis']['weights'])

    invalid_prog = {'ques': ques_invalid_prog}
    return weaver, outputs, invalid_prog

  def _assemble_program(self, image, text, fact, text_feat, tokens, reuse_stack):
    # aliases
    weaver = self.weaver
    executor = self.executor

    # get extent of layout tokens
    max_time, batch_size = tokens.shape
    num_rounds = executor.params['num_rounds']

    outputs = []
    validity = []
    # for visualizing internal nodes
    vis_outputs = {'att': [], 'weights': []}
    for r_id in range(num_rounds):
      layout = tokens[:, r_id]
      invalid_prog = False
      round_id = weaver.batch_input(executor._loom_types['round'], r_id)
      if fact is not None: fact_slice = weaver.slice_fact(fact, round_id)

      # valid layout must contain <eos>. Assembly fails if it doesn't.
      if not np.any(layout == self.EOS_idx): invalid_prog = True

      decode_stack = []
      penult_out = None # penultimate output
      for t_id in range(len(layout)):
        weights = None
        time = weaver.batch_input(executor._loom_types['time'], t_id)
        text_att = weaver.slice_text(text, round_id, time)

        # slice the text feature
        text_feat_slice = weaver.slice_text_feat(text_feat, round_id, time)

        cur_op_id = layout[t_id]
        cur_op_name = self.module_names[cur_op_id]

        # <eos> would mean stop
        if cur_op_id == self.EOS_idx: break

        # insufficient number of inputs
        num_inputs = _module_input_num[cur_op_name]
        if len(decode_stack) < num_inputs:
          invalid_prog = True
          break

        # read the inputs
        inputs = []
        for ii in range(num_inputs):
          arg, arg_type = decode_stack.pop()
          # cannot consume anything but attention
          if arg_type != 'att':
            invalid_prog = True
            break
          inputs.append(arg)

        # switch cases
        if cur_op_name == '_Find':
          out = weaver.find(image, text_att)

        elif cur_op_name == '_Refer':
          # nothing to refer to, wrong program
          if len(reuse_stack) == 0:
            invalid_prog = True
            break

          # if baseline is in the model, take the last output
          if 'baseline' in self.executor.params['model']:
            out = reuse_stack[-1][0]
          else:
            inputs = (text_feat_slice, round_id, reuse_stack)
            out, weights, logits = self.assemble_refer(*inputs)

        elif cur_op_name == '_Exclude':
          # clean up reuse stack to avoid current finds
          neat_stack = reuse_stack.copy()
          for prev_time in range(t_id - 1, 0, -1):
            if neat_stack[-1][-2] == prev_time: neat_stack.pop(-1)

          # nothing to exclude to, wrong program
          if len(neat_stack) == 0:
            invalid_prog = True
            break

          inputs = (text_att, round_id, neat_stack)
          out = self.assemble_exclude(*inputs)
          # collect in reuse stack
          #reuse_stack.append((out, text_att, round_id, r_id, t_id))

        elif cur_op_name == '_Transform':
          out = weaver.transform(inputs[0], image, text_att)

        elif cur_op_name == '_Describe':
          out = weaver.describe(inputs[0], image, text_att)
          # TODO: Do this more carefully!
          penult_out = arg

        elif cur_op_name == '_Exist':
          out = weaver.exist(inputs[0], image, text_att)
          # TODO: Do this more carefully!
          penult_out = arg

        elif cur_op_name == '_Count':
          out = weaver.count(inputs[0], image, text_att)
          # TODO: Do this more carefully!
          penult_out = arg

        elif cur_op_name == '_And':
          out = weaver.and_op(inputs[0], inputs[1])

        elif cur_op_name == '_Diff':
          out = weaver.diff_op(inputs[0], inputs[1])

        # just invert the attention
        elif cur_op_name == '_Not':
          out = weaver.normalize_exclude(inputs[0])

        else:
          print('Current operand not defined: ' + cur_op_name)
          invalid_prog = True

        # collect outputs from all modules (visualize)
        if self.visualize:
          if _module_output_type[cur_op_name] == 'att':
            vis_outputs['att'].append((out, r_id))
          if weights is not None:
            vis_outputs['weights'].extend(weights)

        decode_stack.append((out, _module_output_type[cur_op_name]))

      # Check if only one element is left
      if len(decode_stack) != 1: invalid_prog = True
      # final output is not answer type
      elif decode_stack[0][1] != 'ans': invalid_prog = True

      # record program validity
      validity.append(invalid_prog)

      # if program is invalid, return zeros
      if invalid_prog: outputs.append(weaver.invalid(image))
      else:
        outputs.append(decode_stack[-1][0])

        # if fact is to be used, take the penultimate output
        if executor.params['use_fact']:
          reuse_stack.append((penult_out, fact_slice, round_id, r_id, -1))

    return {'comp': outputs, 'vis': vis_outputs}, reuse_stack, validity
