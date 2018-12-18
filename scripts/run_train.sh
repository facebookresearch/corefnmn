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

DATASET='visdial_v0.9'
DATA_ROOT='data/'

MODEL='nmn-cap-prog-only'
LRATE=0.0001
#--------------------------------------------------------------------

# train the model (example below)
python -u exp_vd/train_sl.py --gpu_id=$GPU_ID --dataset=$DATASET \
    --data_root='data/' --model=$MODEL --batch_size=2 --use_refer \
    --use_fact --generator='mem' --feature_path='data/'\
    --learning_rate=0.0001 --amalgam_text_feats\
    --decoder='disc' --lstm_size 512

# evaluate a checkpoint (example below)
# CHECKPOINT='Tue-07Aug18-16:30:21/model_epoch_000.tmodel'
# PYTHONPATH=. python -u exp_vd/eval_sl.py --gpu_id=$GPU_ID \
#      --checkpoint=$ROOT$CHECKPOINT --test_split='val'

# visualize results (example below)
#ROOT='/checkpoint/skottur/exp_vd/tfmodel/';
###CHECKPOINT='Thu-02Nov17-18:55:08/model_epoch_005.tmodel'; LABEL=''
#CHECKPOINT='Thu-16Nov17-23:20:12/model_epoch_015.tmodel'
#PYTHONPATH=. python -u exp_vd/visualize_sl.py -gpuID $GPU_ID \
#              -checkpoint $ROOT$CHECKPOINT -batchSize 1 -testSplit 'train'
