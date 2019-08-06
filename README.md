# CorefNMN

This repository contains code for the paper:

**Visual Coreference Resolution in Visual Dialog using Neural Module Networks**  
[Satwik Kottur][2], [José M. F. Moura][3], [Devi Parikh][4], [Dhruv Batra][5], [Marcus Rohrbach][6]  
[[PDF][15]] [[ArXiv][1]] [[Code][21]]  
_European Conference on Computer Vision (ECCV), 2018_ 

### Abstract
Visual dialog entails answering a series of questions grounded
in an image, using dialog history as context. In addition to the challenges
found in visual question answering (VQA), which can be seen as oneround
dialog, visual dialog encompasses several more. We focus on one
such problem called visual coreference resolution that involves determining
which words, typically noun phrases and pronouns, co-refer to the
same entity/object instance in an image. This is crucial, especially for
pronouns (e.g., ‘it’), as the dialog agent must first link it to a previous
coreference (e.g., ‘boat’), and only then can rely on the visual grounding
of the coreference ‘boat’ to reason about the pronoun ‘it’. Prior work
(in visual dialog) models visual coreference resolution either (a) implicitly
via a memory network over history, or (b) at a coarse level for the
entire question; and not explicitly at a phrase level of granularity. In
this work, we propose a neural module network architecture for visual
dialog by introducing two novel modules—Refer and Exclude—that perform
explicit, grounded, coreference resolution at a finer word level. 


[![CorefNMN](https://i.imgur.com/OERoNPz.png)][1]
This repository trains our explicit visual coreference model **CorefNMN**
(figure above).


If you find this code useful, consider citing our work:

```
@InProceedings{Kottur_2018_ECCV,
  author    = {Kottur, Satwik and Moura, Jos\'e M. F. and Parikh, Devi and 
               Batra, Dhruv and Rohrbach, Marcus},
  title     = {Visual Coreference Resolution in Visual Dialog using Neural 
               Module Networks},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month     = {September},
  year      = {2018}
}
```


## Setup

The structure for this repository has been inspired from [n2nmn][13] 
github repository.

The code is in Python3 and uses [TensorFlow r1.0][17].
Additionally, it uses [TensorFlow Fold][16] for execution of dynamic networks.
Install instructions for Fold can be found [here][18].

**Note**: Compatibility of Fold has been tested with only TensorFlow r1.0!

Additional python package dependencies can be installed as follows:

```bash
pip install argparse
pip install json
pip install tqdm
pip install numpy
```

Finally, add the current working directory to the python path, i.e., 
`PYTHONPATH=.`.

This repository contains experiments on two datasets: [VisDial v0.9][19]
and [MNIST Dialog][20]. Instructions to train models on each of these datasets 
are given below.

## Experiments on VisDial v0.9 Dataset

### Preprocessing VisDial v0.9
This code has a lot of preprocessing steps, please hold tight!

There are three preprocessing phases. A script for each of these phases has 
been provided in `scripts/` folder.  


**Phase A:** 
In this phase, we will download the data and extract questions and captions
as text files to run parsers (phase B).
The first phase involves running the following command:

```bash
scripts/run_preprocess_1.sh
```
This creates a folder `data/` within which another folder `visdial_v0.9` will 
be created. All our preprocessing steps will operate on files in this folder.

**Phase B:** The second phase runs the [Stanford Parser][9] to acquire weak
program supervision for questions and captions. Follow the steps below:

1. Download the Stanford parser [here][9]. 
2. Next, copy the file `scripts/run_parser.sh` to the same folder as the 
Stanford parser. Ensure that the `VISDIAL_DATA_ROOT` flag in the above script
(after copying) points correctly to the `data/visdial_v0.9` folder.
For example, if you download and extract the Stanford parser in the parent 
folder of this repository, `VISDIAL_DATA_ROOT` should be `../data/visdial_v0.9/`.

Now run the parser using the command `./run_parser.sh` from the parser folder.
This should take about 45-60 min based on your CPU configuration.
Adjust the memory argument in `run_parser.sh` based on your RAM.

**Phase C:** 
For the third phase, first ensure the following:

1. Download the vocabulary file from the original visual dialog codebase
  ([github][7]). The vocabulary for the VisDial v0.9 dataset is 
  [here][8].
1. Save the following files: **`data/visdial_v0.9/vocabulary_layout_4.txt`**
	
	```
	_Find
	_Transform	
	_And
	_Describe
	<eos>
	```
	and **`data/visdial_v0.9/vocabulary_layout_5.txt`**
	
	```
	_Find
	_Transform	
	_And
	_Describe
	_Refer
	<eos>
	```
1. Download the visual dialog data files with weak coreference supervision.
These files have been obtained using off-the-shelf, text-only coreference 
resolution system ([github][10]).
They are available here: [train][11] and [val][12].

Now, run the script for the third phase:
```bash
scripts/run_preprocess_2.sh
```
This will use the output from the Stanford parser, create programs for our 
model, extract vocabulary from train dataset, and finally create image-dialog
database for each split (`train` and `val`) that will be used by our training 
code.


### Extracting Image Features
To extract image features, please follow instructions [here][14].

All instructions for preprocessing are now done! We are set to train visual
dialog models that perform explicit coreference resolution.

### Training
To train a model, look at `scripts/run_train.sh` that shows usages of command 
line flags for the file `exp_vd/train_sl.py`.
Information about these flags can be obtained from `exp_vd/options.py`.

Usage for `exp_vd/eval_sl.py` (evaluating a checkpoint) are also given in 
`scripts/run_train.sh`.


## Experiments on MNIST Dialog Dataset

### Preprocessing MNIST Dialog

In order to preprocess MNIST Dialog dataset, simply run the command:

```bash
scripts/run_preprocess_mnist.sh
```

As before, this will download the dataset and create image-dialog database 
(similar to VisDial v0.9 preprocessing phase C).

Finally save the following at: **`data/mnist/vocabulary_layout_mnist.txt`**

```
_Find
_Transform
_Exist
_Describe
_Refer
_Not
_And
_Count
<eos>
```

Done with preprocessing!

### Training
Training a model is handled by `exp_mnist/train_sl.py`, while 
`exp_mnist/eval_sl.py` handles evaluating a specific checkpoint.
Commandline options are parsed by `exp_mnist/options.py`.

Usage of these scripts is demonstrated by the bash script `scripts/run_mnist.sh`.


### Future Releases
  
- [x] Visualization scripts
- [x] MNIST Experiments
- [ ] Detailed Doc Strings
- [x] Additional installation instructions
- [ ] Include pretrained models

### License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree ([here][22]).
Portions of the source code are from the n2nmn project which is in LICENSE.n2nmn in the root directory of this source tree ([here][23]).

[1]:https://arxiv.org/abs/1809.01816
[2]:https://satwikkottur.github.io/
[3]:https://users.ece.cmu.edu/~moura/
[4]:https://www.cc.gatech.edu/~parikh/
[5]:https://www.cc.gatech.edu/~dbatra/
[6]:http://rohrbach.vision/
[7]:https://github.com/batra-mlp-lab/visdial/
[8]:https://s3.amazonaws.com/visual-dialog/data/v0.9/visdial_params.json
[9]:https://nlp.stanford.edu/software/lex-parser.shtml#Download
[10]:https://github.com/huggingface/neuralcoref
[11]:http://users.ece.cmu.edu/~skottur/datasets/visdial_v0.9/visdial_0.9_train_coref_supervise.json
[12]:http://users.ece.cmu.edu/~skottur/datasets/visdial_v0.9/visdial_0.9_val_coref_supervise.json
[13]:https://github.com/ronghanghu/n2nmn
[14]:https://github.com/ronghanghu/n2nmn#download-and-preprocess-the-data-1
[15]:https://arxiv.org/pdf/1809.01816.pdf
[16]:https://github.com/tensorflow/fold
[17]:https://www.tensorflow.org/versions/r1.0/
[18]:https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/setup.md
[19]:https://visualdialog.org/datahttp://cvlab.postech.ac.kr/research/attmem/
[20]:http://cvlab.postech.ac.kr/research/attmem/
[21]:https://github.com/facebookresearch/corefnmn
[22]:https://github.com/facebookresearch/corefnmn/blob/master/LICENSE
[23]:https://github.com/facebookresearch/corefnmn/blob/master/LICENSE.n2nmn
