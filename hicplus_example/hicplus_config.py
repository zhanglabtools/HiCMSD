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
epochs = 12001
batch_size = 256
learning_rate = 0.00001
down_sample_ratio = 16
max_value = None

subImage_size = 80
step = 35
input_resolution = 10000


