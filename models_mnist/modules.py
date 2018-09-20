"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Module definitions for Loom API.

Explicit visual coreference resolution in visual dialog using neural module
networks. Neural module definitions.

Author: Satwik Kottur
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T
from tensorflow_fold.public import loom

from util.cnn import fc_layer as fc, conv_layer as conv
from util.empty_safe_conv import empty_safe_1x1_conv as _1x1_conv
from util.empty_safe_conv import empty_safe_conv as _conv

def add_spatial_coord_map(image_feat_grid):
  image_feat_shape = tf.shape(image_feat_grid)
  N = image_feat_shape[0]
  # static dimensions
  #H = image_feat_shape[1]
  #W = image_feat_shape[2]
  H, W = image_feat_grid.shape.as_list()[1:3]
  x_map = tf.tile(
    tf.reshape(tf.linspace(-1., 1., W), [1, 1, -1, 1]),
    to_T([N, H, 1, 1]))
  y_map = tf.tile(
    tf.reshape(tf.linspace(-1., 1., H), [1, -1, 1, 1]),
    to_T([N, 1, W, 1]))

  # stop gradient on coords_map (needed to fix the tile grad error on TF 1.0.0)
  coords_map = tf.stop_gradient(tf.concat([x_map, y_map], axis=3))
  image_feat_with_coords = tf.concat([image_feat_grid, coords_map], axis=3)
  # set shapes of the new feature maps
  image_feat_static_shape = image_feat_grid.get_shape().as_list()
  image_feat_static_shape[3] += 2
  image_feat_with_coords.set_shape(image_feat_static_shape)
  image_feat_static_shape[3] = 2
  coords_map.set_shape(image_feat_static_shape)

  return image_feat_with_coords, coords_map
#------------------------------------------------------------------------------

# Simple tensorflow ops as loom ops
class BinaryLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types, op):
    self._op = op
    super(BinaryLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    arg1, arg2 = inputs
    return [self._op(arg1, arg2)]
#------------------------------------------------------------------------------

class UnaryLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types, op):
    self._op = op
    super(UnaryLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, arg):
    return [self._op(arg[0])]
#------------------------------------------------------------------------------

# slice text attention
class SliceTextLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types):
    super(SliceTextLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    text, round_id, time = inputs

    round_squeeze = tf.squeeze(round_id, -1)
    time_squeeze = tf.squeeze(time, -1)

    # select the right round
    shape = text.shape.as_list()
    B = tf.shape(text)[0]
    num_rounds, T, text_dim = shape[1], shape[2], shape[3]
    indices = round_squeeze + num_rounds * tf.range(B)
    # flatten
    result = tf.gather(tf.reshape(text, [-1, T, text_dim]), indices)

    # select the right time
    indices = time_squeeze + T * tf.range(B)
    # flatten
    result = tf.gather(tf.reshape(result, [-1, text_dim]), indices)

    return [result]
#------------------------------------------------------------------------------

# slice answer embeddding
class SliceAnswerLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types):
    super(SliceAnswerLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    answer, round_id = inputs

    round_squeeze = tf.squeeze(round_id, -1)

    # select the right round
    shape = answer.shape.as_list()
    B = tf.shape(answer)[0]
    num_rounds, text_dim = shape[1], shape[2]
    indices = round_squeeze + num_rounds * tf.range(B)

    result = tf.gather(tf.reshape(answer, [-1, text_dim]), indices)

    return [result]
#--------------------------------------------------------------------

# attention weighting
class AttentionWeightLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types):
    super(AttentionWeightLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    vis_att, scalar = inputs

    # simple weighting
    scalar = tf.expand_dims(tf.expand_dims(scalar, -1), -1)
    att_grid = tf.multiply(vis_att, scalar)

    return [att_grid]
#--------------------------------------------------------------------

# identity op to convert types
class IdentityLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types):
    super(IdentityLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    return inputs
#--------------------------------------------------------------------

# normalize and complementary attention
class NormalizeExcludeLoomOp(loom.LoomOp):
  def __init__(self, in_types, out_types):
    super(NormalizeExcludeLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    att_grid = inputs[0]
    # complement the attention
    max_entry = tf.reduce_max(tf.reduce_max(att_grid, 1), 1)
    max_entry = tf.expand_dims(tf.expand_dims(max_entry, 1), 1)
    att_grid = att_grid / max_entry
    att_grid = 1 - att_grid

    # normalize
    norms = tf.reduce_sum(tf.reduce_sum(att_grid, 1), 1)
    norms = tf.expand_dims(tf.expand_dims(norms, 1), 1)
    # cutoff 
    norms = tf.clip_by_value(norms, 1e-6, 1e6)
    att_grid = att_grid / norms

    return [att_grid]
#-------------------------------------------------------------------

class AlignTextLoomOp(loom.LoomOp):
  """
  Takes in two text attention and computes the alignment between them
  Mapping: text_param x text_param -> scalar
  Input:
   text_param: [N, D_txt]
   text_param: [N, D_txt]
  Output:
   scalar: [N, 1]

  Implementation:

  Parameters typically contain:
    map_dim = 1024
    module_scope = alignTextOp
    reuse = True
    scope
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'alignTextOp')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(AlignTextLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
    Inputs:
      image feature for the example
      text attention for all modules for the example
      time id for current module
    """
    text_att1, text_att2, round_id1, round_id2 = inputs

    # text feature dimension, intermediate mapping dimension
    # batch size, image feature height and width
    text_dim = text_att1.shape.as_list()[-1]
    map_dim = self._params['map_dim']
    embed_dim = self._params['text_embed_size']

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        # concat both text attentions, along with round diff (if need be)
        concat_list = [text_att1, text_att2]

        # additional weight for the distance to the past
        if self._params['amalgam_text_feats']:
          round_diff = tf.cast(round_id1 - round_id2, tf.float32)
          concat_list.append(round_diff)

        concat = tf.concat(concat_list, axis=-1)
        # deeper 2 layer align network
        weights = tf.contrib.layers.fully_connected(concat, embed_dim)
        weights = tf.contrib.layers.fully_connected(weights, 1,
                                                    activation_fn=None)

    return [weights]
#--------------------------------------------------------------------

# Modules as Loom Ops
class FindLoomOp(loom.LoomOp):
  """
  Mapping: image_feat_grid x text_param -> att_grid
  Input:
   image_feat_grid: [N, H, W, D_im]
   text_param: [N, D_txt]
  Output:
   att_grid: [N, H, W, 1]

  Implementation:
   1. Elementwise multiplication between image_feat_grid and text_param
   2. L2-normalization
   3. Linear classification

  Parameters typically contain:
    map_dim = 1024
    module_scope = findModule
    reuse = True
    scope
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'find_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(FindLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
    Inputs:
      image feature for the example
      text attention for all modules for the example
      time id for current module
    """
    img_feat, text_att = inputs

    # text feature dimension, intermediate mapping dimension
    # batch size, image feature height and width
    text_dim = text_att.shape.as_list()[-1]
    map_dim = self._params['map_dim']
    N = tf.shape(img_feat)[0]
    H, W = img_feat.shape.as_list()[1:3]

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        # image_feat_mapped has shape [N, H, W, map_dim]
        img_map = _1x1_conv('conv_image', img_feat, output_dim=map_dim)
        # nonlinearity
        img_map = tf.nn.relu(img_map)

        text_map = fc('fc_text', text_att, output_dim=map_dim)
        # nonlinearity
        text_map = tf.nn.relu(text_map)
        text_map = tf.reshape(text_map, [-1, 1, 1, map_dim])

        # interact via element wise map
        eltwise_mult = tf.nn.l2_normalize(img_map * text_map, 3)
        att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

        # softmax
        att_grid_soft = tf.nn.softmax(tf.reshape(att_grid, [-1, H*W]))
        att_grid = tf.reshape(att_grid_soft, [-1, H, W, 1])

    return [att_grid]
#------------------------------------------------------------------------------
class AndLoomOp(loom.LoomOp):
  """
  Mapping: att_grid x att_grid -> att_grid
  Input:
   input_0: [N, H, W, 1]
   input_1: [N, H, W, 1]
  Output:
   att_grid: [N, H, W, 1]

  Implementation:
   Take the elementwise-min

  Parameters typically contain:
    map_dim = 1024
    module_scope = findModule
    reuse = True
    scope
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'and_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(AndLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
      Inputs:
        visual attention outputs
        time id for current module
    """
    input1, input2 = inputs

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        att_grid = tf.minimum(input1, input2)

        # now L1 normalize
        norms = tf.einsum('ijkl->i', att_grid)
        norms = tf.reshape(norms, [-1, 1, 1, 1])
        #norms = tf.tile(tf.reshape(norms, [-1, 1, 1, 1]), [1, H, W, 1])
        # NOTE: if norm is too low, then clip it
        norms = tf.clip_by_value(norms, 1e-6, 1e6)
        att_grid = att_grid / norms

    return [att_grid]
#------------------------------------------------------------------------------

class CountLoomOp(loom.LoomOp):
  """
  Mapping: att_grid -> answer probs
  Input:
   input_0: [N, H, W, 1]
  Output:
   answer_scores: [N, self.num_choices]

  Implementation:
   1. linear transform of the attention map (also including max and min)

  Parameters typically contain:
    map_dim = 1024
    module_scope = count_module
    reuse = True
    scope
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'count_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(CountLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
      Inputs:
        image feature for the example
        text attention for all modules for the example
        time id for current module
    """
    vis_att, img_feat, _ = inputs
    encode_size = self._params['encode_size']

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):

        H, W = img_feat.shape.as_list()[1:3]
        att_all = tf.reshape(vis_att, to_T([-1, H * W]))
        att_min = tf.reduce_min(vis_att, axis=[1, 2])
        att_max = tf.reduce_max(vis_att, axis=[1, 2])
        # att_reduced has shape [N, 3]
        att_concat = tf.concat([att_all, att_min, att_max], axis=1)
        context = fc('fc_scores', att_concat, output_dim=encode_size)

    return [context]
#------------------------------------------------------------------------------

class ExistLoomOp(loom.LoomOp):
  '''
    Mapping: att_grid -> answer probs
    Input:
     att_grid: [N, H, W, 1]
    Output:
     answer_scores: [N, self.num_choices]

    Implementation:
     1. Max-pool over att_grid
     2. a linear mapping layer (without Re_lU)
        Mapping: image_feat_grid x text_param -> att_grid
        Input:
         image_feat_grid: [N, H, W, D_im]
         text_param: [N, D_txt]
        Output:
         att_grid: [N, H, W, 1]

    Parameters typically contain:
      map_dim = 1024
      module_scope = find_module
      reuse = True
      scope
  '''
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'exist_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(ExistLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    '''
      Inputs:
        image feature for the example
        text attention for all modules for the example
        time id for current module
    '''
    vis_att, _, _ = inputs
    encode_size = self._params['encode_size']

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        att_min = tf.reduce_min(vis_att, axis=[1, 2])
        att_avg = tf.reduce_mean(vis_att, axis=[1, 2])
        att_max = tf.reduce_max(vis_att, axis=[1, 2])
        # att_reduced has shape [N, 3]
        att_reduced = tf.concat([att_min, att_avg, att_max], axis=1)
        context = fc('fc_scores', att_reduced, output_dim=encode_size)

    return [context]
#------------------------------------------------------------------------------

class DiffLoomOp(loom.LoomOp):
  '''
    Mapping: att_grid x att_grid -> att_grid
    Input:
     input_0: [N, H, W, 1]
     input_1: [N, H, W, 1]
    Output:
     att_grid: [N, H, W, 1]

    Implementation:
     Take the elementwise diff and lower caps it to zero

    Parameters typically contain:
      map_dim = 1024
      module_scope = find_module
      reuse = True
      scope
  '''
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'diff_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(DiffLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    '''
      Inputs:
        visual attention outputs
        time id for current module
    '''
    input1, input2 = inputs

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        att_grid = tf.maximum(input1 - input2, 0.)

        # now L1 normalize
        norms = tf.einsum('ijkl->i', att_grid)
        norms = tf.reshape(norms, [-1, 1, 1, 1])
        #norms = tf.tile(tf.reshape(norms, [-1, 1, 1, 1]), [1, H, W, 1])
        # NOTE: if norm is too low, then clip it
        norms = tf.clip_by_value(norms, 1e-6, 1e6)
        att_grid = att_grid / norms

    return [att_grid]
#------------------------------------------------------------------------------

class InvalidLoomOp(loom.LoomOp):
  """
    Mapping: returns a context of zeros
    Output:
     context: [N, encode_size] of zeros

    Implementation:
     Take the elementwise-min

    Parameters typically contain:
      map_dim = 1024
      module_scope = find_module
      reuse = True
      scope
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'invalid_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(InvalidLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
      Inputs:
        visual attention outputs
        time id for current module
    """
    img_feat = inputs
    encode_size = self._params['encode_size']

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        N = tf.shape(img_feat)[0]
        context = tf.zeros([N, encode_size], tf.float32)

    return [context]
#------------------------------------------------------------------------------

class DescribeLoomOp(loom.LoomOp):
  """
  Mapping: att_grid -> context vector
  Input:
    input_0: [N, H, W, 1]
  Output:
    answer_scores: [N, outputSize]

  Implementation:
  1. Extract visual features using the input attention map, and
  linear transform to map_dim
  2. linear transform language features to map_dim
  3. Element-wise multiplication of the two, l2_normalize, linear transform.
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'describe_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(DescribeLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
    Inputs:
      output from the previous modules
      image feature for the example
      text attention for all modules for the example
      time id for current module
    """
    vis_att, img_feat, text_att = inputs

    # text feature dimension, intermediate mapping dimension
    # batch size, image feature height and width
    text_dim = text_att.shape.as_list()[-1]
    map_dim = self._params['map_dim']
    encode_size = self._params['encode_size']
    N = tf.shape(img_feat)[0]
    H, W = img_feat.shape.as_list()[1:3]

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        text_map = fc('fc_text', text_att, output_dim=map_dim)
        # nonlinearity
        text_map = tf.nn.relu(text_map)

        # att_feat, att_feat_1 has shape [N, D_vis]
        att_feats = tf.reduce_sum(img_feat * vis_att, axis=[1, 2])
        img_map = tf.reshape(fc('fc_att', att_feats, output_dim=map_dim),
                  [N, map_dim])
        # nonlinearity
        img_map = tf.nn.relu(img_map)

        eltwise_mult = tf.nn.l2_normalize(img_map * text_map, 1)
        context = fc('fc_eltwise', eltwise_mult, output_dim=encode_size)


    return [context]
#------------------------------------------------------------------------------

class TransformLoomOp(loom.LoomOp):
  """
  Mapping: att_grid x text_param -> att_grid
  Input:
   input_0: [N, H, W, 1]
   text_param: [N, D_txt]
  Output:
   att_grid: [N, H, W, 1]

  Implementation:
   1. Extract visual features using the input attention map, and
    linear transform to map_dim
   2. linear transform language features to map_dim
   3. Convolve image features to map_dim
   4. Element-wise multiplication of the three, l2_normalize, linear transform.
  """
  def __init__(self, in_types, out_types, params):
    self._params = params
    self._scope = params.get('scope', 'transform_module')
    self._module_scope = params['module_scope']
    self._reuse = params.get('reuse', None)
    super(TransformLoomOp, self).__init__(in_types, out_types)

  def instantiate_batch(self, inputs):
    """
    Inputs:
      output from the previous modules
      image feature for the example
      text attention for all modules for the example
      time id for current module
    """
    vis_att, img_feat, text_att = inputs

    # text feature dimension, intermediate mapping dimension
    # batch size, image feature height and width
    text_dim = text_att.shape.as_list()[-1]
    map_dim = self._params['map_dim']
    encode_size = self._params['encode_size']
    N = tf.shape(img_feat)[0]
    H, W = img_feat.shape.as_list()[1:3]

    with tf.variable_scope(self._module_scope):
      with tf.variable_scope(self._scope, reuse=self._reuse):
        # image_feat_mapped has shape [N, H, W, map_dim]
        img_map = _1x1_conv('conv_image', img_feat, output_dim=map_dim)
        # nonlinearity
        img_map = tf.nn.relu(img_map)

        text_map = fc('fc_text', text_att, output_dim=map_dim)
        text_map = tf.reshape(text_map, [-1, 1, 1, map_dim])
        # nonlinearity
        text_map = tf.nn.relu(text_map)

        att_feats = tf.reduce_sum(img_feat * vis_att, axis=[1, 2])
        att_map = tf.reshape(fc('fc_att', att_feats, output_dim=map_dim),
                  [N, 1, 1, map_dim])

        # interact via element wise map
        eltwise_mult = tf.nn.l2_normalize(img_map * text_map * att_map, 3)
        att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

        # softmax
        att_grid_soft = tf.nn.softmax(tf.reshape(att_grid, [-1, H*W]))
        att_grid = tf.reshape(att_grid_soft, [-1, H, W, 1])

    return [att_grid]
#------------------------------------------------------------------------------
