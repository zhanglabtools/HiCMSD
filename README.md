# HiCMSD

A package used to improve Hi-C data resolution. 

## Installation

#### Requirement

- Python 3.5+
- numpy 1.15.4 
- scipy 1.1.0 
- scikit-image 0.13.0   scikit-learn 0.20.0
- torch 0.4.1   torchvision 0.2.1
- visdom 0.1.8.5

## Data Processing

As for data processing, preprocess.py in HiCMSD receives sparse matrix, dense matrix or python .npy files as input.    preprocess.py produces .npz files as intermediate files and output four .npz files as training and testing data.

#### Inputs: One or more directories of Hi-C matrix data

- sparse matrix

  **bin 1** \tab **bin 2** \tab **interaction value**

  ```
  10000	10000	797.0
  20000	20000	2.0
  10000	40000	4.0
  40000	40000	9.0
  10000	50000	6.0
  20000	50000	2.0
  40000	50000	15.0
  50000	50000	66.0
  10000	60000	27.0
  20000	60000	4.0
  40000	60000	7.0
  50000	60000	63.0
  60000	60000	480.0
  10000	70000	14.0
  ......
  ```

- dense matrix

  M(i, j) = interaction  value of **bin i** and **bin j**

  ```
  0	0	0	0	0	0	0	0	0	0	......
  0	288	1	0	2	7	16	12	8	6	......
  0	1	1	0	1	0	0	0	0	0	......
  0	0	0	0	0	0	0	0	0	0	......
  0	2	1	0	3	7	8	3	1	1	......
  0	7	0	0	7	21	44	10	4	4	......
  0	16	0	0	8	44	215	94	15	16	......
  0	12	0	0	3	10	94	114	19	14	......
  0	8	0	0	1	4	15	19	19	16	......
  0	6	0	0	1	4	16	14	16	57	......
  .	.	.	.	.	.	.	.	.	.	......
  .	.	.	.	.	.	.	.	.	.	......
  .	.	.	.	.	.	.	.	.	.	......
  ```

- python .npy file

  2-d numpy array

#### Outputs: 

Four .npz files

- train_high.npz
- train_low.npz
- test_high.npz
- test_low.npz

#### Example

There is a data processing script choose chromosomes 1-17 of cell GM12878 as training data and chromosme 18-X of cell GM12878 as testing data.

```python
import sys

# hicmsd_PATH is the path of 'hicmsd' folder
sys.path.append(hicmsd_PATH)

import hicmsd.preprocess as preprocess

high_folders = {'train_cell_dir_list':['../data/GM12878/MAP30_npy/'],
    'train_cell_chr_list':[[str(chr) for chr in range(1,18)]],
    'test_cell_dir_list': ['../data/GM12878/MAP30_npy/'],
    'test_cell_chr_list':[[str(chr) for chr in range(18,22)]+['X']]}
low_folders = {'train_cell_dir_list':['../data/down_GM12878/MAP30_npy/'],
    'train_cell_chr_list':[[str(chr) for chr in range(1,18)]],
    'test_cell_dir_list': ['../data/down_GM12878/MAP30_npy/'],
    'test_cell_chr_list':[[str(chr) for chr in range(18,22)]+['X']]}
data_dir = './data'

preprocess.HiCDataFromFolder(high_folders, low_folders,  data_dir, input_resolution = 10000,  data_type = 'npy',  subImage_size=80,  divide_step=35)
```



## Configuration

A configuration file will function in both training and predicting Hi-C maps.  You only need to change the configuration file to do different experiments.

An example configuration file for HiCMSD is as follows

```python

from hicmsd.hicmsdnet import msdmodel_l30_4last as msdmodel

msdmodel = msdmodel

# training and testing data path
lowdata_dir = '../data_process_example/data/train_low.npz'
highdata_dir = '../data_process_example/data/train_high.npz'
lowtest_dir = '../data_process_example/data/test_low.npz'
hightest_dir = '../data_process_example/data/test_high.npz'

# model save path
modelsave_dir = './model/'

# trained model used to predict Hi-C data
trained_model_dir = './model/pytorch_hg19_model_200'

# log save path
log_dir = './log/log.txt'

# if use gpu, if not, set it to 0
use_gpu = 1
# GPU device index, if not use GPU, set device_ids = []
device_ids = [0]

# training epoch
epochs = 200

# Hyper parameters
batch_size = 128
learning_rate = 0.0005
num_layers = 30
growth_rate = 1
kernel_size = 3
size_diff = 13
dilation_mod = 10

# low resolution Hi-C maps's down rate
down_sample_ratio = 16

# if set a upper bound for training data, for example, if you set it to 100, and then values bigged than 100 in Hi-C maps will be setted to 100
max_value = None

# sample size of your training data
subImage_size = 80

# sample divide step
step = 35

# resolution of Hi-C maps (byte) 
input_resolution = 10000

```



## Train

A script of training HiCMSD is as follows

```python
import sys

# hicmsd_PATH is the path of 'hicmsd' folder
sys.path.append(hicmsd_PATH)

from hicmsd.hicmsdnet import trainMsdnet as tmsd
import hicmsd_config as config

tmsd.trainMsdnet(config)
```

Before you run this script,  you need start the visdom server by command in a terminal to display the training process in your default browser

```shell
python -m visdom.server
```

Then run script file, for example, 'trainScript.py'

```shell
python trainScript.py
```

## Predict

A script of predicting Hi-C maps with HiCMSD is as follows

```python
import sys
# hicmsd_PATH is the path of 'hicmsd' folder
sys.path.append(hicmsd_PATH)

from hicmsd.hicmsdnet import runMsdnet as rmsd
import hicmsd_config as config

# input and output folder path
# input and output are all .npy files

input_dir = '../down_gm12878/MAP30_npy/'
outputdir = '../down_16/GM12878_nomin/hicmsd'

chrlist = [str(i) for i in range(18,23)]
#chrlist.append('X')

for chrN in chrlist:
	inputfile = input_dir + str(chrN) + '_10kb.matrix.npy'
	rmsd.runMsdnet(inputfile, outputfile, chrN, config)
```



## HiCPlus

For HiCPlus, we just use the program provided by the authors of HiCPlus [HiCPlus Code](https://github.com/zhangyan32/HiCPlus). However, we ensemble configuration to a file just like what we do in HiCMSD.

### configuration

```python
from hicmsd.hicplus import hicplusmodel as hicplusmodel

hicplusmodel = hicplusmodel
lowdata_dir = '../data_process_example/data/train_low.npz'
highdata_dir = '../data_process_example/data/train_high.npz'
lowtest_dir = '../data_process_example/data/test_low.npz'
hightest_dir = '../data_process_example/data/test_high.npz'
modelsave_dir = './model/'
trained_model_dir = './model/pytorch_hg19_model_12000'
log_dir = './log/log.txt'
use_gpu = 1
device_ids = [0]
epochs = 12000
batch_size = 256
learning_rate = 0.00001
down_sample_ratio = 16
max_value = None

subImage_size = 80
step = 35
input_resolution = 10000

```

### train

```python
import sys

# hicmsd_PATH is the path of 'hicmsd' folder
sys.path.append(hicmsd_PATH)

#import msdnet.trainMsdnet as msdtrain
from hicmsd.hicplus import trainhicplus as tmsd
import hicplus_config as config

tmsd.trainhicplus(config)

```

### predict

```python
import sys

# hicmsd_PATH is the path of 'hicmsd' folder
sys.path.append(hicmsd_PATH)

from hicmsd.hicplus import runhicplus as rhs
import hicplus_config as config

# down matrices folder path
input_dir = '../down_gm12878/MAP30_npy/'
# output matrices folder path
outputdir = '../down_16/GM12878_nomin/hicplus'

chrlist = [str(i) for i in range(18,23)]
#chrlist.append('X')

for chrN in chrlist:
	inputfile = input_dir + str(chrN) + '_10kb.matrix.npy'
	rhs.runhicplus(inputfile, outputdir, chrN, config)
```



## Gaussian Smoothing

```python
import sys
# hicmsd_PATH is the path of 'hicmsd' folder
sys.path.append(hicmsd_PATH)
import numpy as np

from hicmsd import Gaussian_tools as gt

# first experiment with cell gm12878 data
low_dir = '../down_16/5kb/5kb_MAP30_npy/'
output_dir = '../down_16/5kb/gaussian/'

down_rate = 16
test_list = [str(chr) for chr in range(18,23)]
#test_list.append('X')
file_name_end = '_5kb.matrix.npy'
sigma_vector = np.arange(4,5, 1)


for sigma in sigma_vector:
    for i in test_list:
        print('sigma = %d, chr %s'%(sigma, i))
        file_path = low_dir + str(i) + file_name_end
        mat_topredict = np.load(file_path)
        mat_topredict = mat_topredict * down_rate
        predict_mat = gt.Gaussian_filter(mat_topredict, sigma)
        np.save( output_dir+str(i)+'_5kb.matrix.npy', predict_mat)
```







