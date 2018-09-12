# Bash script to run pre-processing for 
# CorefNMN: Explicit visual coreference resolution for visual dialog 
# using neural module networks (part 2/2)


DATA_ROOT='data/visdial_v0.9/'
DTYPES=('train' 'val')
OBJS=('ques' 'cap')

# convert the stanford parser output to nmn outputs
for DTYPE in "${DTYPES[@]}"
do
  for OBJ in "${OBJS[@]}"
  do
    # setup paths
    LOAD_PATH=${DATA_ROOT}"visdial_0.9_"$DTYPE"_"${OBJ?}
    PROGRAM_PATH=${DATA_ROOT}"visdial_0.9_"$DTYPE"_"${OBJ?}"_att.pgm"

    python util/compress_parser_trees.py --parser_file=${LOAD_PATH?}'.sps'
    python util/parse.py $LOAD_PATH'_compress.sps' > $PROGRAM_PATH
    python util/convert_nmn_layouts.py \
        --nmn_file=${PROGRAM_PATH?} \
        --visdial_file=${DATA_ROOT}"visdial_0.9_"$DTYPE".json" \
        --prog_vocab_file=${DATA_ROOT}"vocabulary_layout_4.txt"
  done
done

# construct vocabulary and gather glove features
python util/collect_glove_features.py \
       --vocab_file=${DATA_ROOT}"visdial_params.json" \
       --save_path=${DATA_ROOT}"vocabulary_vd.txt"

# build image dataset and pack train | val dataset
for DTYPE in "${DTYPES[@]}"
do
  # setup paths
  QUES_PATH=${DATA_ROOT}"visdial_0.9_"$DTYPE"_ques_att.layout.npy"
  CAP_PATH=${DATA_ROOT}"visdial_0.9_"$DTYPE"_cap_att.layout.npy"
  IMG_FMT="data/images/"$DTYPE"2014/COCO_"$DTYPE"2014_%012d.jpg"
  FEAT_PATH="data/resnet_152/"$DTYPE"2014/COCO_"$DTYPE"2014_%012d.npy"
  COREF_PATH=${DATA_ROOT}"visdial_0.9_"$DTYPE"_coref_supervise.json"
  VISDIAL_PATH=${DATA_ROOT}"visdial_0.9_"$DTYPE".json"
  SAVE_PATH=${DATA_ROOT}"imdb_"$DTYPE".npy"
  VOCAB_FILE=${DATA_ROOT}"vocabulary_vd.txt"

  python util/build_imdb.py --ques_prog_file=$QUES_PATH \
                            --cap_prog_file=$CAP_PATH \
                            --image_path_format=$IMG_FMT \
                            --feature_path=$FEAT_PATH \
                            --coreference_file=$COREF_PATH \
                            --visdial_file=$VISDIAL_PATH \
                            --save_path=${SAVE_PATH} \
                            --vocab_file=$VOCAB_FILE
done
