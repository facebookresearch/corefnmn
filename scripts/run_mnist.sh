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

# train the model (example below)
# python -u exp_mnist/train_eval.py --gpu_id=$GPU_ID --dataset=$DATASET \
#     --data_root='data/' --model=$MODEL --batch_size=30 --use_refer \
#     --use_fact --amalgam_text_feats --remove_aux_find

# evaluate a checkpoint (example below)
#ROOT='checkpoints/'
ROOT='/coc/pskynet1/skottur3/tfmodel/'
CHECKPOINT='Wed-24Oct18-16:08:02/model_epoch_156.tmodel'
python -u exp_mnist/eval_sl.py --gpu_id=$GPU_ID \
  --checkpoint=$ROOT$CHECKPOINT --test_split='test'

# visualize results (example below)
#ROOT='/checkpoint/skottur/exp_vd/tfmodel/';
#PYTHONPATH=. python -u exp_vd/visualize_sl.py -gpuID $GPU_ID \
#              -checkpoint $ROOT$CHECKPOINT -batchSize 1 -testSplit 'train'

# python util/visualize_mnist.py --batch_path=batch_model.npy
