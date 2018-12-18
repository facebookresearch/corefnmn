'''Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.
'''

GPU_ID=$CUDA_VISIBLE_DEVICES

if [ -z "$GPU_ID" ] 
then
  GPU_ID=-1
fi

DATASET='mnist'
DATA_ROOT='data/'

MODEL='nmn-cap-prog-only'
LRATE=0.0001
#--------------------------------------------------------------------

# train the model (example below)
# python -u exp_mnist/train_sl.py --gpu_id=$GPU_ID --dataset=$DATASET \
#     --data_root='data/' --model=$MODEL --batch_size=30 --use_refer \
#     --use_fact --amalgam_text_feats --remove_aux_find

# evaluate a checkpoint (example below)
# test_split could be 'valid' or 'test'
ROOT='checkpoints/'
CHECKPOINT='folder/model_epoch_000.tmodel'
python -u exp_mnist/eval_sl.py --gpu_id=$GPU_ID \
  --checkpoint=$ROOT$CHECKPOINT --test_split='test'

# visualize results (example below)
#ROOT='/checkpoint/skottur/exp_vd/tfmodel/';
#PYTHONPATH=. python -u exp_vd/visualize_sl.py -gpuID $GPU_ID \
#              -checkpoint $ROOT$CHECKPOINT -batchSize 1 -testSplit 'train'
