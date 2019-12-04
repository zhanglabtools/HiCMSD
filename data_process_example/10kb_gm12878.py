import sys

sys.path.append('E:/Users/bzhang/hic/')

import hicmsd.preprocess as preprocess

high_folders = {'train_cell_dir_list':['E:/Users/bzhang/hic/juicer_run/data/GM12878/MAP30_npy/'],
    'train_cell_chr_list':[[str(chr) for chr in range(1,18)]],
    'test_cell_dir_list': ['E:/Users/bzhang/hic/juicer_run/data/GM12878/MAP30_npy/'],
    'test_cell_chr_list':[[str(chr) for chr in range(18,22)]+['X']]}
low_folders = {'train_cell_dir_list':['E:/Users/bzhang/hic/juicer_run/data/down_GM12878/MAP30_npy/'],
    'train_cell_chr_list':[[str(chr) for chr in range(1,18)]],
    'test_cell_dir_list': ['E:/Users/bzhang/hic/juicer_run/data/down_GM12878/MAP30_npy/'],
    'test_cell_chr_list':[[str(chr) for chr in range(18,22)]+['X']]}
data_dir = './data'

preprocess.HiCDataFromFolder(high_folders, low_folders,  data_dir, input_resolution = 10000,  data_type = 'npy',  subImage_size=80,  divide_step=35)
