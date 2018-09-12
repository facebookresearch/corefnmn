# CorefNMN

This repository contains code for the paper:

**Visual Coreference Resolution in Visual Dialog using Neural Module Networks**  
[Satwik Kottur][2], [José M. F. Moura][3], [Devi Parikh][4], [Dhruv Batra][5], [Marcus Rohrbach][6]  
[[PDF][15]] [[ArXiv][1]]  
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


## Setup Instructions

The code structure for this repository has been inspired from [n2nmn][13] 
github repository.

1. The current code uses tensorflow fold and Tensorflow v1.0.

1. Additional packages like `json`, `h5py`, etc., can be installed using `pip`.

1. Add the present working directory to the python path, i.e., `PYTHONPATH=.`


### Preprocessing Instructions
This code has a lot of preprocessing steps, please hold tight!

There are three preprocessing phases. A script for each of these phases has 
been provided in `scripts/` folder.  


A. The first phase involves running the following command:

```bash
scripts/run_preprocess_1.sh
```
This creates a folder `data/` within which another folder `visdial_v0.9` will 
be created. All our preprocessing steps will operate on files in this folder.

B. The second phase is to run the Stanford Parser to acquire program 
supervision for questions and captions. Follow the steps below.

1. Download the Stanford parser [here][9]. 
2. Next, copy the file `scripts/run_parser.sh` to the same folder as the 
Stanford parser. Ensure that the `VISDIAL_DATA_ROOT` flag in the above script
(after copying) points correctly to the `data/visdial_v0.9` folder.
For example, if you download and extract the Stanford parser in the parent 
folder of this repository, `VISDIAL_DATA_ROOT` should be `../data/visdial_v0.9/`.
Before running any of these, ensure the following:

Now run the parser using the command `./run_parser.sh` from the parser folder.
This should take about 45-60 min based on your CPU configuration.
Feel free to adjust the memory argument in `run_parser.sh` to suit your system.

C. For the third phase, ensure the following before running the corresponding
script:

1. Download the vocabulary file from the original visual dialog codebase
  ([github][7]). Specifically, the vocabulary for the VisDial v0.9 dataset is 
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
1. Download the visual dialog data files with coreference supervision. These
files have been obtained using off-the-shelf, text-only coreference resolution
system ([github][10]). The files are available at -- [train][11] and [val][12].

### Extracting Image Features
To extract the image features, please follow instructions [here][14].

All instructions for preprocessing are now done! We are set to train visual
dialog models that perform explicit coreference resolution.

### Training
To train a model, please look at `run_train.sh` script that contains all common
arguments.
Similar examples have been given for evaluating a checkpoint.

### TODOs
  
- [ ] Visualization scripts
- [ ] MNIST Experiments
- [ ] Detailed Doc Strings and instructions
- [ ] Additional installation instructions
- [ ] Include pretrained models


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
