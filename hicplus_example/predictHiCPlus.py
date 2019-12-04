

import sys

sys.path.append('/home/tiger/HiC/HiCMSD/')

from hicmsd.hicplus import runhicplus as rhs
import hicplus_config as config

input_dir = 'E:/Users/bzhang/biodata/Generate_data_20181102/down_gm12878/MAP30_npy/'
outputdir = 'E:/Users/bzhang/biodata/down_16/GM12878_nomin/hicplus'

chrlist = [str(i) for i in range(18,23)]
#chrlist.append('X')

for chrN in chrlist:
	inputfile = input_dir + str(chrN) + '_10kb.matrix.npy'
	rhs.runhicplus(inputfile, outputdir, chrN, config)
