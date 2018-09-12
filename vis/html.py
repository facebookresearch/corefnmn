"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.
"""

import pdb
import sys
import numpy as np

class HTML():
  def __init__(self, cols, header_file='util/jquery_header.html'):
    self.template = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"+\
            "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">' +\
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'

    self.template += '<style>'+\
            'table#t01{width:100%; background-color:#fff}'\
            +'table#t01 tr:nth-child(odd){background-color:#ddd;}'\
            +'table#t01 tr:nth-child(even){background-color:#fff;}'\
            +'table#t01 tr td tr:nth-child(odd){background-color:#ddd;}'\
            +'table#t01 tr td tr:nth-child(even){background-color:#fff;}'\
            +'table#t01 th{background-color:black;color:white}'+\
            '</style>'
    self.colors = ['maroon', 'red', 'purple', 'fuchsia',
            'green', 'lime', 'olive', 'yellow',
            'navy', 'blue', 'teal', 'aqua', 'orange']

    with open(header_file, 'r') as file_id: self.template += file_id.read()

    self.template += '</head><body><table id ="t01">'
    self.end = '</table></body></html>'
    self.content = ''
    self.row_content = '<tr>'+'<td valign="top">%s</td>'*cols+'</tr>'
    self.span_first_content = '<tr>'+'<td valign="top" rowspan="%s">%s</td>'+\
                '<td valign="top">%s</td>' * (cols-1)+'</tr>'
    self.span_other_content = '<tr>'+ '<td valign="top">%s</td>'*(cols-1)\
                      +'</tr>'
    self.att_template = '<mark style="background-color:rgba(255,0,0,%f)"> %s </mark>|'
    self.img_template = '<img src="%s" height="%d" width="%d"></img>'

    # creating table
    self.num_rows = None
    self.num_cols = cols

  # Add a new row
  def add_spanning_row(self, mega_row, *entries):
    # if first element is list, take it
    if type(entries[0]) == list: entries = entries[0]

    for index, ii in enumerate(entries):
      if len(ii) != self.num_cols - 1:
        print('Warning: Incompatible entries.\n_taking needed!')

      if len(ii) < self.num_cols - 1: # add 'null'
        for jj in range(self.num_cols - 1 - len(entries)):
          entries[index].append('NULL')

    num_rows = len(entries)
    content = (num_rows, mega_row)+tuple(entries[0])
    new_row = self.span_first_content % content
    for ii in range(1, num_rows):
      new_row += self.span_other_content % tuple(entries[ii])

    # Add new_row to content
    self.content += new_row

  # Add a new row
  def add_row(self, *entries):
    # if first element is list, take it
    if type(entries[0]) == list: entries = entries[0]

    if len(entries) != self.num_cols:
      print('Warning: Incompatible number of entries.\n_taking needed!')

    if len(entries) < self.num_cols: # add 'null'
      for ii in range(self.num_cols - len(entries)):
        entries.append('NULL')

    new_row = self.row_content % tuple(entries)
    # Add new_row to content
    self.content += new_row

  # setting the title
  def set_title(self, titles):
    new_titles = []
    for ii in titles: new_titles.append('<strong>%s</strong>' % ii)
    self.add_row(new_titles)

  # coloring text
  def get_colored_text(self, text, group_id=None):
    ''' If group id is None, pick a random color '''
    if group_id is None: color = self.colors[1]
    else: color = self.colors[group_id % len(self.colors)]

    return '<b><font color="%s">%s</font></b>' % (color, text)

  # render and save page
  def save_page(self, file_path):
    # allow new page and tab space
    self.content = self.content.replace('\n', '</br>')
    self.content = self.content.replace('\t', '&nbsp'*10)
    page_content = self.template + self.content + self.end
    with open(file_path, 'w') as file_id: file_id.write(page_content)
    print('Written page to: %s' % file_path)

  # Return the string for an image
  def link_image(self, img_path, caption=None, height=100):
    # No caption provided
    if caption == None: return self.img_template % (img_path, height, height)

    string = 'Caption: %s</br>' % caption
    return string + (self.img_template % (img_path, height, height))

  # add table with question encoding
  def add_question_attention(self, question, program, att):
    table = '<table class="heat-map" id="heat-map-3"><thead><tr><th></th>'
    row = ''.join(['<th>%s</th>' % ii for ii in program])
    row += '</tr></thead><tbody>'
    table += row

    for ii in range(len(question)):
      table += '<tr class="stats-row"><td class="stats-title">%s</td>'\
                                % question[ii]
      table += ''.join(['<td>%2d</td>' % att[ii, jj] \
                        for jj in range(len(program))])
      table += '</tr>'

    table += '</tbody></table>'
    return table

  # add history attention
  def add_history_attention(self, att_wt, att_labels = None):
    num_ques = att_wt.size
    if att_labels is None:
      titles = ['Cap']
      titles.extend(['%02d' % ii for ii in range(1, num_ques)])
    else: titles = att_labels

    max_att = np.max(att_wt)
    string = ''
    for ii in range(0, num_ques):
      if ii % 6 == 0: string += '\n'
      string += self.att_template % (att_wt[ii]/max_att, titles[ii])

    return string
