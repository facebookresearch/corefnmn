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
python -u exp_mnist/train_sl.py --gpu_id=$GPU_ID --dataset=$DATASET \
    --data_root='data/' --model=$MODEL --batch_size=30 --use_refer \
    --use_fact --amalgam_text_feats --use_batch_norm --remove_aux_find

# evaluate a checkpoint (example below)
ROOT='checkpoints/'
# CHECKPOINT='Sat-22Sep18-23:52:08/model_epoch_000.tmodel'
# python -u exp_mnist/eval_sl.py --gpu_id=$GPU_ID \
#       --checkpoint=$ROOT$CHECKPOINT --test_split='valid'

# visualize results (example below)
#ROOT='/checkpoint/skottur/exp_vd/tfmodel/';
#PYTHONPATH=. python -u exp_vd/visualize_sl.py -gpuID $GPU_ID \
#              -checkpoint $ROOT$CHECKPOINT -batchSize 1 -testSplit 'train'
