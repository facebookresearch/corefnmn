"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script that converts Stanford parser outputs to neural module
network layout outputs.
"""

import argparse
import copy
import json
import os
import pdb
import re
import sys
import sexpdata

import numpy as np

from models_vd.assembler import Assembler
from tqdm import tqdm as progressbar


def extract_parse(p):
  """Given string, extracts a parse.
  """

  if isinstance(p, sexpdata.Symbol):
    return p.value()
  elif isinstance(p, int):
    return str(p)
  elif isinstance(p, bool):
    return str(p).lower()
  elif isinstance(p, float):
    return str(p).lower()
  return tuple(extract_parse(q) for q in p)


def parse_tree(p):
  if "'" in p:
    p = "none"
  parsed = sexpdata.loads(p)
  extracted = extract_parse(parsed)
  return extracted

parse2module_dict = {'find': '_Find',
                     'relate': '_Transform',
                     'and': '_And',
                     'is': '_Describe', # All the top modules go to '_Describe'
                     'describe': '_Describe'
                    }


def flatten_layout(parse):
  # Postorder traversal to generate Reverse Polish Notation (RPN)
  if isinstance(parse, str):
    return [parse2module_dict[parse]]

  RPN = []
  head = parse[0]
  body = parse[1:]
  module = parse2module_dict[head]
  for m in body:
    RPN += flatten_layout(m)
  RPN += [module]
  return RPN


def extract_set(params):
  # assembler to look for incorrect programs
  assembler = Assembler(params.prog_vocab_file)

  # manual correction to layouts
  layout_correct = {('_Find', '_Transform', '_And', '_Describe')
                      :['_Find', '_Transform', '_Describe'],
                    ('_Transform', '_Describe')
                      :['_Find', '_Transform', '_Describe'],
                    ('_Transform', '_Transform', '_And', '_Describe')
                      :['_Find', '_Transform', '_Transform', '_Describe'],
                    ('_Describe',)
                      :['_Find', '_Describe'],
                    ('_Transform', '_Find', '_And', '_Describe')
                      :['_Find', '_Transform', '_Describe']}

  with open(params.nmn_file) as f:
    # drop the spans
    read_layouts = [re.sub(r'\[\d*,\d*\]', '', ll) for ll in f.readlines()]
    layouts = [flatten_layout(parse_tree(ll)) for ll in read_layouts]
    layouts = [layout_correct.get(tuple(ii), tuple(ii)) for ii in layouts]

  with open(params.nmn_file) as f:
    # extracting spans as well
    lines = [ii for ii in f.readlines()]
    attentions = []
    for index, ii in enumerate(lines):
      layout = layouts[index]
      # extract the spans
      matches = re.findall('(\w\w)\[(\d*),(\d*)\]', ii)

      # match module with attention, if present
      att = []
      for token in layout:
        candidates = []
        if token == '_Find':
          candidates = [jj for jj in matches if jj[0] == 'nd']

        if token == '_Transform':
          candidates = [jj for jj in matches if jj[0] == 'te']

        if token == '_Describe':
          candidates = [jj for jj in matches
                        if jj[0] != 'te' or jj[0] != 'nd']

        if len(candidates) >= 1:
          att.append((int(candidates[0][1]), int(candidates[0][2])))
          matches.remove(candidates[0])
        else:
          att.append((0, 0))

      # record attentions and layouts
      attentions.append(att)

  # correct the layouts according to the above dictionary
  layouts = [layout_correct.get(tuple(ii), ii) for ii in layouts]
  layout_set = {tuple(l) for l in layouts}

  print('Found %d unique layouts' % len(layout_set))
  for l in layout_set:
    print(' ', ' '.join(list(l)))

  # check whether the layout is valid
  for l in layout_set:
    batch = assembler.module_list2tokens(l, T=20)
    validity, error = assembler.sanity_check_program(batch)

    if not validity:
      raise Exception('invalid expr:' + str(l) + ' ' + error)

  # read the original data path
  with open(params.visdial_file, 'r') as file_id:
    vd_data = json.load(file_id)

  # question id to layout dictionary
  if params.question:
    qid2layout_dict = {}
    for datum in progressbar(vd_data['data']['dialogs']):
      img_id = datum['image_id']
      for r_id, round_datum in enumerate(datum['dialog']):
        q_id = img_id * 10 + r_id
        q_layout = layouts[round_datum['question']]
        # record
        qid2layout_dict[q_id] = q_layout

    np.save(params.save_path, np.array(qid2layout_dict))
  else:
    np.save(params.save_path, np.array(layouts))
  print('Saving to: ' + params.save_path)

  save_file_att = params.save_path.replace('.layout', '.attention')
  print('Saving (att) to: ' + save_file_att)
  np.save(save_file_att, np.array(attentions))

  set_layout_length = [len(l) for l in layouts]
  return set_layout_length


def main(FLAGS):
  # check if it is question or caption
  FLAGS.question = 'ques' in FLAGS.nmn_file
  FLAGS.save_path = FLAGS.nmn_file.replace('pgm', 'layout')

  print('Saving at: %s' % FLAGS.save_path)
  layout_length = extract_set(FLAGS)

  print('Program length distribution:')
  print(np.unique(layout_length, return_counts=True))


if __name__ == '__main__':
  title = 'Converting parser outputs to neural module network programs'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--nmn_file', required=True,
                      help='Neural Module file path')
  parser.add_argument('--visdial_file', required=True,
                      help='Path to the original visdial file')
  parser.add_argument('--prog_vocab_file', required=True,
                      help='Path to program vocabulary file for the assembler')
  FLAGS = parser.parse_args()

  main(FLAGS)
