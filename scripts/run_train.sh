# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# Portions of the source code are from the n2nmn project which
# notice below and in LICENSE.n2nmn in the root directory of
# this source tree.
# 
# Copyright (c) 2017, Ronghang Hu
# All rights reserved.

GPU_ID=$CUDA_VISIBLE_DEVICES

DATASET='visdial_v0.9'
DATA_ROOT='data/'

MODEL='nmn-cap-prog-only'
LRATE=0.0001
#--------------------------------------------------------------------

# train the model (example below)
python -u exp_vd/train_sl.py --gpu_id=$GPU_ID --dataset=$DATASET \
    --data_root='data/' --model=$MODEL --batch_size=10 --use_refer \
    --use_fact --generator='mem' --feature_path='data/'\
    --learning_rate=0.0001 --amalgam_text_feats\
    --decoder='disc' --lstm_size 512

# evaluate a checkpoint (example below)
# CHECKPOINT='checkpoints/Fri-02Aug19-19:42:47/model_epoch_000.tmodel'
# python -u exp_vd/eval_sl.py --gpu_id=$GPU_ID \
#      --checkpoint=$CHECKPOINT --test_split='val'

# visualize results (example below)
# CHECKPOINT='checkpoints/Mon-05Aug19-18:27:15/model_epoch_001.tmodel'
# python -u exp_vd/visualize_sl.py --gpu_id $GPU_ID \
#     --checkpoint=$CHECKPOINT --batch_size 1 --test_split='train'

# python -u vis/visualize_dialogs.py --batch_path=$CHECKPOINT'.50_batches.npy'\
#   --text_vocab_path='data/visdial_v0.9/vocabulary_vd.txt'\
  --prog_vocab_path='data/visdial_v0.9/vocabulary_layout_5.txt'\
  --num_examples=10
