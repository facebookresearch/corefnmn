# Bash script to run pre-processing for 
# CorefNMN: Explicit visual coreference resolution for visual dialog 
# using neural module networks (part 1/2)

# create the necessary folders
DATA_ROOT='data/visdial_v0.9/'
mkdir -p ${DATA_ROOT}

# download the files
TRAIN_LINK=https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_train.zip
VAL_LINK=https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_val.zip

# download and unzip
wget -P ${DATA_ROOT} ${TRAIN_LINK?}
wget -P ${DATA_ROOT} ${VAL_LINK?}

TRAIN_DATA='visdial_0.9_train.zip'
VAL_DATA='visdial_0.9_val.zip'

unzip ${DATA_ROOT}${TRAIN_DATA?} -d $DATA_ROOT
unzip ${DATA_ROOT}${VAL_DATA?} -d $DATA_ROOT

# preprocess
python scripts/dataset_to_text.py --data_file=${DATA_ROOT}"visdial_0.9_train.json"
python scripts/dataset_to_text.py --data_file=${DATA_ROOT}"visdial_0.9_val.json"

echo 'Proceed with running Stanford parser step. See README.md for details.'
