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

# Bash script to run pre-processing for 
# CorefNMN: Explicit visual coreference resolution for visual dialog 
# using neural module networks
# MNIST experiments

# create the necessary folders
DATA_ROOT='data/mnist/'
mkdir -p ${DATA_ROOT}

# download the files
# DATA_LINK=http://cvlab.postech.ac.kr/research/attmem/data/dataset.json
# IMG_LINK=http://cvlab.postech.ac.kr/research/attmem/data/imgs.zip
# 
# # download and unzip
# wget -P ${DATA_ROOT} ${DATA_LINK?}
# wget -P ${DATA_ROOT} ${IMG_LINK?}
# 
# unzip "${DATA_ROOT}imgs.zip" -d ${DATA_ROOT}

# extract the image dialog database
python util/build_imdb_mnist.py \
    --json_path="${DATA_ROOT}dataset.json"\
    --image_root="${DATA_ROOT}imgs/"\
    --vocab_save_path=${DATA_ROOT}"vocabulary_mnist.txt"\
    --answers_save_path=${DATA_ROOT}"answers_mnist.txt"\
    --imdb_save_path=${DATA_ROOT}\
    --mean_save_path=${DATA_ROOT}"train_image_mean.npy"
