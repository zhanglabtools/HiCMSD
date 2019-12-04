import sys

sys.path.append('/home/tiger/HiC/HiCMSD/')

from hicmsd.hicmsdnet import trainMsdnet as tmsd
import hicmsd_config as config


tmsd.trainMsdnet(config)

