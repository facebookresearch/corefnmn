# script to visualize intermediate outputs from a trained checkpoint
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import pdb, sys, argparse, os, json
from time import gmtime, strftime
from tqdm import tqdm as progressbar
from exp_mnist import options
from util import support

# read command line options
parser = argparse.ArgumentParser();
parser.add_argument('-checkpoint', required=True, \
                            help='Checkpoint to load the models');
parser.add_argument('-batchSize', type=int, default=10, \
                            help='Batch size for evaluation / visualization');
parser.add_argument('-testSplit', default='valid', \
                            help='Which split to run evaluation on');
parser.add_argument('-gpuID', type=int, default=0)

try: args = vars(parser.parse_args());
except (IOError) as msg: parser.error(str(msg));

# set the cuda environment variable for the gpu to use
if args['gpuID'] >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpuID']);
    print(os.environ['CUDA_VISIBLE_DEVICES'])
else: os.environ['CUDA_VISIBLE_DEVICES'] = '';

# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))

from models_mnist.assembler import Assembler
from models_mnist.model import NMN3Model
from util.mnist_train.data_reader import DataReader
from util.metrics import computeMetrics, ExpSmoothing

# setting random seeds
np.random.seed(1234);
tf.set_random_seed(1234);

# read the train args from checkpoint
paramPath = args['checkpoint'].replace('.tmodel', '_params.json');
with open(paramPath, 'r') as fileId: savedArgs = json.load(fileId);
savedArgs.update(args);
args = savedArgs;
args['preloadFeats'] = False;
args['superviseAttention'] = False;
args['useFact'] = args.get('useFact', False);
print('Current model: ' + args['model'])

# Data files
imdbPathVal = os.path.join(args['dataRoot'],'imdb/imdb_%s.npy'%args['testSplit']);
imdbPathVal = imdbPathVal.replace('.npy', '_%s.npy' % args['dataLabel']);

# assembler
assembler = Assembler(args['progVocabPath']);

# dataloader for val
inputDict = {'path':imdbPathVal, 'shuffle':False, 'onePass':True, 'args':args,\
             'assembler': assembler, 'useCount': False, 'fetchOptions': True};
valLoader = DataReader(inputDict);

# The model for training
evalParams = args.copy();
evalParams['useGTProg'] = False; # for training
evalParams['encDropout'] = False;
evalParams['decDropout'] = False;
evalParams['decSampling'] = False; # do not sample, take argmax

# for models trained later
if 'numRounds' not in evalParams:
    evalParams['numRounds'] = valLoader.batchLoader.numRounds;

# model for evaluation
# create another assembler of caption
assemblers = {'ques': assembler, 'cap': Assembler(args['progVocabPath'])};
model = NMN3Model(evalParams, assemblers);

# Load snapshot
print('Loading checkpoint from: %s' % args['checkpoint'])
snapshot_saver = tf.train.Saver(max_to_keep=None);  # keep all snapshots
snapshot_saver.restore(sess, args['checkpoint']);

print('Evaluating on %s' % args['testSplit'])
ansMatches = []; progMatches = [];
totalIter = int(valLoader.batchLoader.numInst / args['batchSize']);
maxIters = 100; curIter = 0;
toSave = {'output': [], 'batch': []};

for batch in progressbar(valLoader.batches(), total=totalIter):
    _, outputs = model.runVisualizeIteration(batch, sess);

    toSave['output'].append(outputs);
    toSave['batch'].append(batch);

    # debug -- also compute the ranks during visualization
    #ranks.append(batchRanks);

    curIter += 1;
    if curIter >= maxIters: break;

# save the output + batch
batchPath = args['checkpoint'] + '.100_batches.npy';
print('Printing the batches: ' + batchPath)
support.saveBatch(toSave, batchPath);

# debug evaluate
#metrics = computeMetrics(np.hstack(ranks));
