"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Visualizing the dialog output from the model.

Explicit visual coreference resolution in visual dialog using neural module
networks.

Author: Satwik Kottur
"""

import argparse
import numpy as np
import sys
import h5py
import json
import os
from vis import html
from util import support

# PIL
from PIL import Image
import requests
from io import BytesIO
from util import support
from tqdm import tqdm as progressbar
from skimage import io, transform

def main(args):
  titles = ['Image', 'Answers', 'Predictions', 'Modules', 'Attention']

  # load the batch
  data = np.load(load_path)[()]
  batch, outputs = data['batch'], data['output']

  # load dictionary
  with open(args.text_vocab_path, 'r') as file_id:
    word2ind = {word.strip('\n'): ind
                for ind, word in enumerate(file_id.readlines())}
    ind2word = {ind: word for word, ind in word2ind.items()}

  # get the program dictionary
  with open(args.prog_vocab_path, 'r') as file_id:
    word2ind_prog = {word.strip('\n'): ind
                     for ind, word in enumerate(file_id.readlines())}
    ind2word_prog = {ind: word for word, ind in word2ind_prog.items()}

  stringify = lambda vector: ' '.join([ind2word[w] for w in vector])
  stringify_prog = lambda vector: ' '.join([ind2word_prog[w] for w in vector])

  # Get html related info
  page = html.HTML(len(titles))
  page.set_title(titles)
  template = 'Q%d: %s\nA [GT]: %s\nP [GT]: %s\nP: %s'
  pred_template = 'GT Rank: %d\n_top-5: \n%s'

  # saving intermediate outputs
  end_prog_token = word2ind_prog['<eos>']
  server_save = './attention/%d_%d_%d_%d.png'
  local_save = os.path.join(root, './attention/%d_%d_%d_%d.png')

  for ii in progressbar(range(args.num_examples)):
    # image
    img_name = '/'.join(batch[ii]['img_path'][0].split('/')[-2:])
    # read image
    image = io.imread(img_template + img_name)

    # caption
    if batch[ii]['cap_len'].ndim == 2:
      cap_len = batch[ii]['cap_len'][0]
      cap_string = stringify(batch[ii]['cap'][0, :cap_len])
    else:
      cap_len = batch[ii]['cap_len'][0]
      cap_string = stringify(batch[ii]['cap'][0, :cap_len])

    span_content = page.link_image(img_template + img_name, cap_string, 400)
    # decide length based on first appearance of 14 <eos>
    if 'pred_tokens_cap' in outputs[ii]:
      caption_prog = outputs[ii]['pred_tokens_cap']
      prog_len = np.where(caption_prog[:, 0] == end_prog_token)[0][0]
      cap_tokens = [ind2word[w] for w in batch[ii]['cap'][0, :cap_len]]
      prog_tokens = [ind2word_prog[w] for w in caption_prog[:prog_len, 0]]
      att = 100 * outputs[ii]['attention_cap'][:, :, 0, 0].transpose()
      word_att_str = page.add_question_attention(cap_tokens, prog_tokens, att)

      # caption module outputs
      stack = outputs[ii]['intermediates'][0]
      cap_stack = [datum for datum in stack if datum[0] == 'cap']
      string = {'c_1':'', 'c_2':''}
      for _, step, _, attention in cap_stack:
        # reshape and renormalize
        att = attention[:, :, 0]
        att_image = support.get_blend_map(image, att)
        att_image = Image.fromarray(np.uint8(att_image))
        att_image = att_image.resize((200, 200))
        att_image.save(local_save % (2, ii, 0, step), 'png')
        # caption first row
        string['c_1'] += page.link_image(server_save % (2, ii, 0, step))

        att = attention[:, :, 0]
        att_image = support.interpolate_attention(image, att)
        #att_image = support.get_blend_map(image, att)
        att_image = Image.fromarray(np.uint8(att_image))
        att_image = att_image.resize((200, 200))
        att_image.save(local_save % (3, ii, 0, step), 'png')
        # caption second row
        string['c_2'] += page.link_image(server_save % (3, ii, 0, step))

      # add the neural module visualization for captions
      span_content += '\n'.join(['', string['c_1'], string['c_2'], word_att_str])

    ques_content = []
    for jj in range(10):
      row_content = []
      # question
      ques_len = batch[ii]['ques_len'][jj]
      ques_string = stringify(batch[ii]['ques'][:ques_len, jj])

      # answer
      ans_len = batch[ii]['ans_len'][jj]
      ans_in = stringify(batch[ii]['ans_in'][jj, :ans_len])
      ans_out = stringify(batch[ii]['ans_out'][jj, :ans_len])

      # program
      gt_prog_str = stringify_prog(batch[ii]['gt_layout'][:, jj])
      cur_prog = outputs[ii]['pred_tokens'][:, jj]
      prog_pred = stringify_prog(outputs[ii]['pred_tokens'][:, jj])

      print_slot = (jj, ques_string, ans_in, gt_prog_str, prog_pred)
      row_content.append(template % print_slot)

      # get predictions
      sort_arg = np.argsort(outputs[ii]['scores'][jj])[::-1][:args.top_options]
      gt_score = outputs[ii]['scores'][jj][batch[ii]['gt_ind'][jj]]
      gt_rank = np.sum(outputs[ii]['scores'][jj] > gt_score) + 1
      options = [stringify(batch[ii]['opt_in'][kk][jj]) for kk in sort_arg]
      row_content.append(pred_template % (gt_rank, '\n'.join(options)))

      # visualizing intermediate outputs for each question
      stack = outputs[ii]['intermediates'][0]
      ques_stack = [datum for datum in stack
                    if (datum[0] == 'ques') and (datum[2] == jj)]
      string = {'q_1':'', 'q_2':''}
      for _, step, _, attention in ques_stack:
        # reshape and renormalize
        att = attention[:, :, 0]
        #att_image = support.interpolate_attention(image, att)
        att_image = support.get_blend_map(image, att)
        att_image = Image.fromarray(np.uint8(att_image))
        att_image = att_image.resize((200, 200))
        att_image.save(local_save % (0, ii, jj, step), 'png')
        # string for first row
        string['q_1'] += page.link_image(server_save % (0, ii, jj, step))

        att = attention[:, :, 0]
        att_image = support.interpolate_attention(image, att)
        #att_image = support.get_blend_map(image, att)
        att_image = Image.fromarray(np.uint8(att_image))
        att_image = att_image.resize((200, 200))
        att_image.save(local_save % (1, ii, jj, step), 'png')
        # string for second row
        string['q_2'] += page.link_image(server_save % (1, ii, jj, step))

        # if refer module, add weights
        if ind2word_prog[cur_prog[step]] == '_Refer':
          wt_stack = outputs[ii]['intermediates'][1]
          cur_wt = [datum for datum in wt_stack if datum[0] == jj]
          assert (len(cur_wt) == 1), 'Weights over history do not sum to one'
          wts = cur_wt[0][1]
          wt_labels = cur_wt[0][2]
          if len(wts) > 0:
            string['q_1'] = page.add_history_attention(wts, wt_labels)
            string['q_1'] += ('\n' + string['q_1'])

      row_content.append('\n'.join(['', string['q_1'], string['q_2']]))

      # decide length based on first appearance of 14 <eos>
      ques_prog = outputs[ii]['pred_tokens'][:, jj]
      prog_len = np.where(ques_prog == end_prog_token)[0][0]

      ques_tokens = [ind2word[w] for w in batch[ii]['ques'][:ques_len, jj]]
      prog_tokens = [ind2word_prog[w] for w in ques_prog[:prog_len]]
      att = 100 * outputs[ii]['attention'][:, :, jj, 0].transpose()
      string = page.add_question_attention(ques_tokens, prog_tokens, att)
      row_content.append(string)
      ques_content.append(row_content)

    # Add the span row
    page.add_spanning_row(span_content, ques_content)

  # render page and save
  page.save_page(args.save_path)

if __name__ == '__main__':
  # read command line arguments
  title = 'Visualizing dialog by creating a HTML page.'
  parser = argparse.ArgumentParser(description=title)

  parser.add_argument('--batch_path', default='logs/sample_run_batches.npy',
                      help='Path to batches saved by visualize_sl.py')
  parser.add_argument('--text_vocab_path', default='data/vocab_vd.txt',
                      help='Text vocabulary to decode sentence outputs')
  parser.add_argument('--prog_vocab_path', default='data/vocab_layout.txt',
                      help='Program vocabulary to decode program outputs')
  parser.add_argument('--save_path', default='vis/sample_run_examples.html',
                      help='Save the HTML file that visualizes examples')
  parser.add_argument('--image_root', default='vis/images/',
                      help='Path to the images to load in HTML')

  parser.add_argument('--num_examples', default=50, type=int,
                      help='Number of examples to visualize')
  parser.add_argument('--top_options', default=5, type=int,
                      help='Number of top ranked options to show')

  args = parser.parse_args()
  main(args)
