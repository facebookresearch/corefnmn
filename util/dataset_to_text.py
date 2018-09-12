"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Script to flatten the dataset for Stanford parser.
"""

import argparse
import json
import sys
from unidecode import unidecode
from tqdm import tqdm as progressbar

def clean_non_ascii(text):
  """Method to clean up and convert non-ascii to unicode.
  """

  try:
    text = text.decode('ascii')
  except:
    # Contains non-ascii symbols
    # Check if it needs to be converted to unicode
    try:
      text = unicode(text, encoding = 'utf-8')
    except:
      pass
    text = unidecode(text)

  return text


def main(args):
  # reading data
  print('Reading from: ' + args.data_file)
  with open(args.data_file, 'r') as file_id:
    data = json.load(file_id)

  # open a text file to write the questions
  save_path = args.data_file.replace('.json', '_ques_flat.txt')
  print('Saving to: ' + save_path)
  with open(save_path, 'w') as file_id:
    for ques in progressbar(data['data']['questions']):
      file_id.write(clean_non_ascii(ques) + ' ?\n')

  # open a text file to write the captions
  save_path = args.data_file.replace('.json', '_cap_flat.txt')
  print('Saving to: ' + save_path)
  with open(save_path, 'w') as file_id:
    captions = [ii['caption'] for ii in data['data']['dialogs']]

    for cap in captions:
      file_id.write(clean_non_ascii(cap) + ' .\n')


if __name__ == '__main__':
  title = 'Flattening the dataset to a text file'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--data_file', required=True,
                      help='Data file path')
  args = parser.parse_args()

  main(args)
