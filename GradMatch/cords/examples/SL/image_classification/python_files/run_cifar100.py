"""
This run file shows how to use the default SL train loop provided in CORDS and 
run it with default arguments using the configuration files provided in CORDS.
"""

import sys
sys.path.append("./cords")

#Train full and GradMatch
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
#CORDS comes with some predefined configuration files that mentiones the format of 
config_file = "configs/SL/config_full_cifar100.py"
#config_file = "configs/SL/config_gradmatchpb_cifar100.py"

## Train KAKURENBO
#from kakurenbo_train_sl import TrainClassifier
#config_file = "configs/SL/config_full_cifar100_kakurenbo.py"


config_data = load_config_data(config_file)
classifier = TrainClassifier(config_data)

classifier.cfg.dss_args.fraction = 0.3
classifier.cfg.dss_args.select_every = 5
classifier.cfg.train_args.device = 'cuda'
classifier.cfg.train_args.print_every = 1
classifier.cfg.is_reg = True


classifier.train()
