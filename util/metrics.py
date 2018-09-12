"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Methods to compute metrics given the list of ranks.

Author: Satwik Kottur
"""

import numpy as np

# static list of metrics
metric_list = ['r1', 'r5', 'r10', 'mean', 'mrr']
# +1 - greater the better
# -1 - lower the better
trends = [1, 1, 1, -1, -1, 1]


def evaluate_metric(ranks, metric):
  """
  Args:
    ranks: List of ranks
    metric: Name of the metric to be computed

  Returns:
    Appropriate evaluation of the metric
  """

  if metric == 'r1':
    ranks = ranks.reshape(-1)
    return 100 * np.sum(ranks <= 1)/float(ranks.shape[0])
  if metric == 'r5':
    ranks = ranks.reshape(-1)
    return 100 * np.sum(ranks <= 5)/float(ranks.shape[0])
  if metric == 'r10':
    ranks = ranks.reshape(-1)
    return 100 * np.sum(ranks <= 10)/float(ranks.shape[0])
  if metric == 'mean':
    ranks = ranks.reshape(-1)
    return np.mean(ranks)
  if metric == 'mrr':
    ranks = ranks.reshape(-1)
    return np.mean(1/ranks)


def compute_metrics(ranks, silent=False):
  """Compute standard metrics, given the ranks.

  Args:
    ranks: List of ranks
    silent: To decide the verbosity

  Returns:
    results: Dictionary of metrics
  """

  results = {metric: evaluate_metric(ranks, metric) for metric in metric_list}
  # pretty print metrics
  if not silent:
    pretty_print_metrics(results)

  return results


def pretty_print_metrics(results):
  """Pretty print the metrics given as a dictionary.
  """

  # pretty print metrics
  print('\n')
  for metric in metric_list: print('\t%s : %.3f' % (metric, results[metric]))


class ExponentialSmoothing:
  """Class responsible to exponentially smooth and track losses.
  """

  def __init__(self):
    self.value = None
    self.blur = 0.95
    self.op = lambda x, y: self.blur * x + (1 - self.blur) * y

  # add a new value
  def report(self, new_val):
    if self.value == None:
      self.value = new_val
    else:
      self.value = {key: self.op(value, new_val[key])
                    for key, value in self.value.items()}
    return self.value
