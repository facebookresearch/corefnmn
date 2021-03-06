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

VISDIAL_DATA_PATH=../../visdial-release/visdial-nmn/data/visdial_v0.9/

DTYPES=('train' 'val')
OBJS=('ques' 'cap')

scriptdir=`dirname $0`

for DTYPE in "${DTYPES[@]}"
do
  for OBJ in "${OBJS[@]}"
  do
    # setup load and save paths
    FILE_PATH=${VISDIAL_DATA_PATH?}"visdial_0.9_"${DTYPE?}"_"${OBJ?}"_flat.txt"
    SAVE_PATH=${VISDIAL_DATA_PATH?}"visdial_0.9_"$DTYPE"_"${OBJ?}

    java -mx8g -cp "$scriptdir/*:" \
      edu.stanford.nlp.parser.lexparser.LexicalizedParser \
      -outputFormat "penn" \
      -outputFormatOptions "stem,collapsedDependencies,includeTags" \
      -sentences newline \
      edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
      $FILE_PATH 1> $SAVE_PATH'.sps'
  done
done
