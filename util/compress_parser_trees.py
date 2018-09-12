"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script to read the data files and emit sentences as a file.
"""

import argparse
import sys


def main(args):
  print('Reading : ' + args.parser_file)
  with open(args.parser_file, 'r') as file_id:
    lines = [ii.strip('\n') for ii in file_id.readlines()]

  # compress trees from multiple lines -> single line
  trees = []
  cur_tree = ''
  for line in lines:
    if line == '':
      trees.append(cur_tree)
      cur_tree = ''
    else:
      cur_tree += line

  # write back to another file
  save_path = args.parser_file.replace('.sps', '_compress.sps')
  print('Saving to: ' + save_path)
  with open(save_path, 'w') as file_id:
    file_id.write('\n'.join(trees))


if __name__ == '__main__':
  title = 'Restructure Stanford Parser to a single line'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--parser_file', required=True,
                      help='Stanford parser output file')
  args = parser.parse_args()

  main(args)
