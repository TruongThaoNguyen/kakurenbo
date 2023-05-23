import torch
import argparse
from utils.dataset_utils2 import download_dataset_to_ssd
import horovod.torch as hvd

from tqdm import tqdm

import os
import zipfile
import os.path
import re
import math
import time
from mpi4py import MPI
    
# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--local-train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--local-val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')

if __name__ == '__main__':
    args = parser.parse_args()
    hvd.init()
    rank = hvd.rank()
    
    print ("Run with arguments:") if hvd.rank() == 0 else None
    for key, value in args._get_kwargs():
        if value is not None:
            print(value,key) if hvd.rank() == 0 else None
            
    local_train_folder = download_dataset_to_ssd(4,args.train_dir,args.local_train_dir)
    MPI.COMM_WORLD.Barrier()
    local_val_folder = download_dataset_to_ssd(4,args.val_dir,args.local_val_dir)
    MPI.COMM_WORLD.Barrier()
    #hostname = MPI.Get_processor_name()
    #command = "ls " + str(local_train_folder)
    #print(rank, hostname, command)
    #os.system(command)