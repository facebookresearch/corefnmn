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

# Script that contains methods for processing answers and questions
# coding: utf-8
import re, pdb
from unidecode import unidecode

# Method used to clean up and convert non ascii to unicode
def clean_non_ascii(text):
  try:
    text = text.decode('ascii')
  except:
    # Contains non-ascii symbols
    # Check if it needs to be converted to unicode
    try: text = unicode(text, encoding = 'utf-8')
    except: pass
    text = unidecode(text)

  return text
