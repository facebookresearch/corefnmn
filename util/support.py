"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script with supporting functions for the main train program.
"""

import os
import sys
import subprocess

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform, filters
from PIL import Image


def last_relevant(output, length):
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.shape[2])
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant


# blending attention map with an image
# source: 
# github.com/abhshkdz/neural-vqa-attention/blob/master/attention_visualization.ipynb
def get_blend_map(img, att_map, blur=True, overlap=True):
  #att_map = softmax(att_map)
  #print(np.min(att_map), np.max(att_map))
  # range it from -1 to 1
  #att_map -= att_map.min()
  if att_map.max() != 0: att_map /= att_map.max()
  image_size = img.shape[:2]
  att_map = transform.resize(att_map, image_size, order = 3)
  if blur:
    att_map = filters.gaussian(att_map, 0.05*max(img.shape))
    #att_map -= att_map.min()
    att_map /= att_map.max()
  cmap = plt.get_cmap('jet')
  att_map_v = cmap(att_map)
  att_map_v = np.delete(att_map_v, 3, 2)
  att_map_v *= 255
  if overlap:
    #vis_im = att_map_v * att_map + (1-att_reshaped)*all_white
    #vis_im = att_map_v*im + (1-att_reshaped)*all_white
    att_map = 1*(1-att_map**0.7).reshape(att_map.shape + (1,))*img \
            + (att_map**0.7).reshape(image_size + (1,)) * att_map_v
  return att_map


# pretty prints dictionary
def pretty_print_dict(parsed):
  max_len = max([len(ii) for ii in parsed.keys()])
  fmt_string = '\t%' + str(max_len) + 's : %s'
  print('Arguments:')
  #for key_pair in parsed.items(): print(fmt_string % key_pair)
  # sort in alphabetical order
  keys = [ii for ii in parsed.keys()]
  keys.sort()
  for key in keys: print(fmt_string % (key, parsed[key]))


# softmax
# correct solution:
def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum() # only difference


# interpolate attention
def interpolate_attention(im, att):
  # steps:
  # 1. reshape the attention to image size (with cubic)
  #soft_att = softmax(att)
  soft_att = att
  att_reshaped = transform.resize(soft_att, im.shape[:2], order=3)
  att_reshaped /= np.max(att_reshaped)
  att_reshaped = att_reshaped[..., np.newaxis]

  # heat map
  #cmap = plt.get_cmap('jet')
  #vis_im = cmap(att_reshaped)
  #vis_im *= (255 if im.dtype == np.uint8 else 1)

  # white + image
  all_white = np.ones_like(im) * (255 if im.dtype == np.uint8 else 1)
  vis_im = att_reshaped * im + (1 - att_reshaped) * all_white
  vis_im = vis_im.astype(im.dtype)
  return vis_im


# shuffling data for image - caption to train alignment
#class Shuffler:
#    def __init__(self, batch_size):
#        assert batch_size > 1, 'Batch size should be greater than 1'
#        self.batch_size = batch_size
def shuffle(arg_list, batch_size):
  # get the batch size
  #batch_size = arg_list[0].shape[0] // 10
  # first five remain the same
  indices = np.random.randint(0, batch_size-1, 10*batch_size)

  for ii in range(batch_size):
    indices[10*ii:10*ii+5] = ii
    diag = indices[10*ii+5:10*ii+10]
    diag[diag >= ii] += 1
    indices[10*ii+5:10*ii+10] = diag

  shuffled = [None for args in arg_list]
  for ii, args in enumerate(arg_list):
    assert batch_size == args.shape[0]
    shuffled[ii] = args[indices]

  return shuffled


# loading an image and converting to numpy
def load_image(file_name) :
  img = Image.open(file_name)
  img.load()
  data = np.asarray(img, dtype="int32")

  return data


# temporary launching of evaluation job (slurm)
def launch_evaluation_job(output_path, checkpoint):
  script_path = 'run_slurm_eval_mnist.sh'

  # read and edit accordingly
  with open(script_path, 'r') as file_id:
    template = file_id.read();

  # write a temporary script, run and remove
  temp_path = script_path.replace('.sh', '_temp.sh');
  with open(temp_path, 'w') as file_id:
    file_id.write(template % (output_path, checkpoint));

  subprocess.call('sbatch %s' % temp_path, shell=True);
