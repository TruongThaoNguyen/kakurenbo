import logging
import wandb
import os
import os.path as osp
import sys
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ray import tune
from cords.selectionstrategies.helpers.ssl_lib.param_scheduler import scheduler as step_scheduler
from cords.utils.data.data_utils.weightedsubset import WeightedSubset, WeightedSubsetX
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, AdaptiveRandomDataLoader, StochasticGreedyDataLoader,\
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader, WeightedRandomDataLoader, MILODataLoader, SELCONDataLoader
from cords.utils.data.dataloader.SL.nonadaptive import FacLocDataLoader, MILOFixedDataLoader
from cords.utils.data.datasets.SL import gen_dataset
#from cords.utils.data.datasets.SL.kakurenbo_builder import gen_datasetX as gen_dataset #NguyenTT
from kakurenbo_utils.weight_sampler import HiddenSamplerUniform as dsampler
from cords.utils.models import *
from cords.utils.data.data_utils.collate import *
import pickle
from datetime import datetime

class TrainClassifier:
    def __init__(self, config_file_data):
        self.cfg = config_file_data
        results_dir = osp.abspath(osp.expanduser(self.cfg.train_args.results_dir))
        
        if self.cfg.dss_args.type in ['StochasticGreedyExploration', 'WeightedRandomExploration', 'SGE', 'WRE']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.kw)
        elif self.cfg.dss_args.type in ['MILO']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.gc_ratio) + "_" + str(self.cfg.dss_args.kw)
        else:
            subset_selection_name = self.cfg.dss_args.type
            
        all_logs_dir = os.path.join(results_dir, 
                                    self.cfg.setting,
                                    self.cfg.dataset.name,
                                    subset_selection_name,
                                    self.cfg.model.architecture,
                                    str(self.cfg.dss_args.fraction),
                                    str(self.cfg.dss_args.select_every),
                                    str(self.cfg.train_args.run))

        os.makedirs(all_logs_dir, exist_ok=True)
        # setup logger
        plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                            datefmt="%m/%d %H:%M:%S")
        now = datetime.now()
        current_time = now.strftime("%y/%m/%d %H:%M:%S")
        self.logger = logging.getLogger(__name__+"  " + current_time)
        self.logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.INFO)
        self.logger.addHandler(s_handler)
        f_handler = logging.FileHandler(os.path.join(all_logs_dir, self.cfg.dataset.name + "_" +
                                                     self.cfg.dss_args.type + ".log"), mode='w')
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False

    
    """
    ############################## Loss Evaluation ##############################
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.cfg.train_args.device), \
                                  targets.to(self.cfg.train_args.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    ############################## Model Creation ##############################
    """

    def create_model(self):
        if self.cfg.model.architecture == 'RegressionNet':
            model = RegressionNet(self.cfg.model.input_dim)
        elif self.cfg.model.architecture == 'ResNet18':
            model = ResNet18(self.cfg.model.numclasses)
            if self.cfg.dataset.name in ['cifar10', 'cifar100', 'tinyimagenet']:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
        elif self.cfg.model.architecture == 'ResNet101':
            model = ResNet101(self.cfg.model.numclasses)
            if self.cfg.dataset.name in ['cifar10', 'cifar100', 'tinyimagenet']:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
        elif self.cfg.model.architecture == 'MnistNet':
            model = MnistNet()
        elif self.cfg.model.architecture == 'ResNet164':
            model = ResNet164(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet':
            model = MobileNet(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNetV2':
            model = MobileNetV2(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet2':
            model = MobileNet2(output_size=self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'HyperParamNet':
            model = HyperParamNet(self.cfg.model.l1, self.cfg.model.l2)
        elif self.cfg.model.architecture == 'ThreeLayerNet':
            model = ThreeLayerNet(self.cfg.model.input_dim, self.cfg.model.num_classes, self.cfg.model.h1, self.cfg.model.h2)
        elif self.cfg.model.architecture == 'LSTM':
            model = LSTMClassifier(self.cfg.model.numclasses, self.cfg.model.wordvec_dim, \
                 self.cfg.model.weight_path, self.cfg.model.num_layers, self.cfg.model.hidden_size)
        else:
            raise(NotImplementedError)
        model = model.to(self.cfg.train_args.device)
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """

    def loss_function(self):
        if self.cfg.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        elif self.cfg.loss.type == "MeanSquaredLoss":
            criterion = nn.MSELoss()
            criterion_nored = nn.MSELoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):
        if self.cfg.optimizer.type == 'sgd':
            if ('ResNet' in self.cfg.model.architecture) and ('lr1' in self.cfg.optimizer.keys()) and ('lr2' in self.cfg.optimizer.keys()) and ('lr3' in self.cfg.optimizer.keys()):
                optimizer = optim.SGD( [{"params": model.linear.parameters(), "lr": self.cfg.optimizer.lr1},
                                        {"params": model.layer4.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.layer3.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.layer2.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.layer1.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.conv1.parameters(), "lr": self.cfg.optimizer.lr3}],
                                    lr=self.cfg.optimizer.lr,
                                    momentum=self.cfg.optimizer.momentum,
                                    weight_decay=self.cfg.optimizer.weight_decay,
                                    nesterov=self.cfg.optimizer.nesterov)
            else:
                optimizer = optim.SGD(model.parameters(),
                                    lr=self.cfg.optimizer.lr,
                                    momentum=self.cfg.optimizer.momentum,
                                    weight_decay=self.cfg.optimizer.weight_decay,
                                    nesterov=self.cfg.optimizer.nesterov)
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif self.cfg.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg.optimizer.lr)

        if self.cfg.scheduler.type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.cfg.scheduler.T_max)
        elif self.cfg.scheduler.type == 'cosine_annealing_WS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=self.cfg.scheduler.T_0,
                                                                   T_mult=self.cfg.scheduler.T_mult)
        elif self.cfg.scheduler.type == 'linear_decay':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=self.cfg.scheduler.stepsize, 
                                                        gamma=self.cfg.scheduler.gamma)
        elif self.cfg.scheduler.type == 'multistep':    
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.scheduler.milestones,
                                                             gamma=self.cfg.scheduler.gamma)
        elif self.cfg.scheduler.type == 'cosine_annealing_step':
            scheduler = step_scheduler.CosineAnnealingLR(optimizer, max_iteration=self.cfg.scheduler.max_steps)
        else:
            scheduler = None
        return optimizer, scheduler

    @staticmethod
    def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing

    @staticmethod
    def save_ckpt(state, ckpt_path):
        torch.save(state, ckpt_path)

    @staticmethod
    def load_ckpt(ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def count_pkl(self, path):
        if not osp.exists(path):
            return -1
        return_val = 0
        file = open(path, 'rb')
        while(True):
            try:
                _ = pickle.load(file)
                return_val += 1
            except EOFError:
                break
        file.close()
        return return_val

    def train(self, **kwargs):
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        logger = self.logger
        if ('trainset' in kwargs) and ('validset' in kwargs) and ('testset' in kwargs) and ('num_cls' in kwargs):
            trainset, validset, testset, num_cls = kwargs['trainset'], kwargs['validset'], kwargs['testset'], kwargs['num_cls']
        else:
            #logger.info(self.cfg)
            if self.cfg.dataset.feature == 'classimb':
                trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,  #NguyenTT
                                                                self.cfg.dataset.name,
                                                                self.cfg.dataset.feature,
                                                                classimb_ratio=self.cfg.dataset.classimb_ratio, dataset=self.cfg.dataset)
            else:
                trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir, #NguyenTT
                                                                self.cfg.dataset.name,
                                                                self.cfg.dataset.feature, dataset=self.cfg.dataset)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size

        batch_sampler = lambda _, __ : None
        drop_last = False
        if self.cfg.dss_args.type in ['SELCON']:
            drop_last = True
            assert(self.cfg.dataset.name in ['LawSchool_selcon', 'Community_Crime'])
            if self.cfg.dss_arg.batch_sampler == 'sequential':
                batch_sampler = lambda dataset, bs : torch.utils.data.BatchSampler(
                    torch.utils.data.SequentialSampler(dataset), batch_size=bs, drop_last=True
                )   # sequential
            elif self.cfg.dss_arg.batch_sampler == 'random':
                batch_sampler = lambda dataset, bs : torch.utils.data.BatchSampler(
                    torch.utils.data.RandomSampler(dataset), batch_size=bs, drop_last=True
                )   # random


        if self.cfg.dataset.name == "sst2_facloc" and self.count_pkl(self.cfg.dataset.ss_path) == 1 and self.cfg.dss_args.type == 'FacLoc':
            self.cfg.dss_args.type = 'Full'
            file_ss = open(self.cfg.dataset.ss_path, 'rb')
            ss_indices = pickle.load(file_ss)
            file_ss.close()
            trainset = torch.utils.data.Subset(trainset, ss_indices)

        if 'collate_fn' not in self.cfg.dataloader.keys():
            collate_fn = None
        else:
            collate_fn = self.cfg.dataloader.collate_fn

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, sampler=batch_sampler(trainset, trn_batch_size),
                                                  shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)
        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size, sampler=batch_sampler(validset, val_batch_size),
                                                shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)

        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size, sampler=batch_sampler(testset, tst_batch_size),
                                                 shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)
	
        train_eval_loader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size * 20, sampler=batch_sampler(trainset, trn_batch_size),
                                                  shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)

        val_eval_loader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size * 20, sampler=batch_sampler(validset, val_batch_size),
                                                shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)

        test_eval_loader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size * 20, sampler=batch_sampler(testset, tst_batch_size),
                                                 shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)
						 
        substrn_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = []
        trn_acc = list()
        val_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        best_acc = list()
        fraction_size = []
        curr_best_acc = 0
        subtrn_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])

        # Checkpoint file
        checkpoint_dir = osp.abspath(osp.expanduser(self.cfg.ckpt.dir))
        
        if self.cfg.dss_args.type in ['StochasticGreedyExploration', 'WeightedRandomExploration', 'SGE', 'WRE']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.kw)
        elif self.cfg.dss_args.type in ['MILO']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.gc_ratio) + "_" + str(self.cfg.dss_args.kw)
        else:
            subset_selection_name = self.cfg.dss_args.type
        
        ckpt_dir = os.path.join(checkpoint_dir, 
                                self.cfg.setting,
                                self.cfg.dataset.name,
                                subset_selection_name,
                                self.cfg.model.architecture,
                                str(self.cfg.dss_args.fraction),
                                str(self.cfg.dss_args.select_every),
                                str(self.cfg.train_args.run))
                                
        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model = self.create_model()
        if self.cfg.train_args.wandb:
            wandb.watch(model)

        # model1 = self.create_model()

        #Initial Checkpoint Directory
        init_ckpt_dir = os.path.abspath(os.path.expanduser("checkpoints"))
        os.makedirs(init_ckpt_dir, exist_ok=True)
        
        model_name = ""
        for key in self.cfg.model.keys():
            if r"/" not in str(self.cfg.model[key]):
                model_name += (str(self.cfg.model[key]) + "_")

        if model_name[-1] == "_":
            model_name = model_name[:-1]
            
        if not os.path.exists(os.path.join(init_ckpt_dir, model_name + ".pt")):
            ckpt_state = {'state_dict': model.state_dict()}
            # save checkpoint
            self.save_ckpt(ckpt_state, os.path.join(init_ckpt_dir, model_name + ".pt"))
        else:
            checkpoint = torch.load(os.path.join(init_ckpt_dir, model_name + ".pt"))
            model.load_state_dict(checkpoint['state_dict'])

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        
        if self.cfg.scheduler.type == "cosine_annealing_step":
            if self.cfg.dss_args.type == "Full":
                self.cfg.scheduler.max_steps = math.ceil(len(list(dataloader.batch_sampler)) * self.cfg.train_args.num_epochs)
            else:
                self.cfg.scheduler.max_steps = math.ceil(len(list(dataloader.subset_loader.batch_sampler)) * self.cfg.train_args.num_epochs)
                 # * self.cfg.dss_args.fraction)

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if 'collate_fn' not in self.cfg.dss_args:
                self.cfg.dss_args.collate_fn = None

        if self.cfg.dss_args.type in ['GradMatch', 'GradMatchPB', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = GradMatchDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                             batch_size=self.cfg.dataloader.batch_size,
                                             shuffle=self.cfg.dataloader.shuffle,
                                             pin_memory=self.cfg.dataloader.pin_memory,
                                             collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['GLISTER', 'GLISTER-Warm', 'GLISTERPB', 'GLISTERPB-Warm']:
            """
            ############################## GLISTER Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            dataloader = GLISTERDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                           batch_size=self.cfg.dataloader.batch_size,
                                           shuffle=self.cfg.dataloader.shuffle,
                                           pin_memory=self.cfg.dataloader.pin_memory,
                                           collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['CRAIG', 'CRAIG-Warm', 'CRAIGPB', 'CRAIGPB-Warm']:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = CRAIGDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                         batch_size=self.cfg.dataloader.batch_size,
                                         shuffle=self.cfg.dataloader.shuffle,
                                         pin_memory=self.cfg.dataloader.pin_memory,
                                         collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['Random', 'Random-Warm']:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = RandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory, 
                                          collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['AdaptiveRandom', 'AdaptiveRandom-Warm']:
            """
            ############################## AdaptiveRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = AdaptiveRandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,
                                            collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['MILOFixed', 'MILOFixed-Warm']:
            """
            ############################## MILOFixed Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = MILOFixedDataLoader(trainloader, self.cfg.dss_args, logger,
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory, 
                                          collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['WeightedRandomExploration', 'WeightedRandomExploration-Warm', 'WRE', 'WRE-Warm']:
            """
            ############################## WeightedRandomDataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = WeightedRandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,
                                            collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['StochasticGreedyExploration', 'StochasticGreedyExploration-Warm', 'SGE', 'SGE-Warm']:
            """
            ############################## StochasticGreedyDataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = StochasticGreedyDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,
                                            collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['MILO']:
            """
            ############################## MILODataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = MILODataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,
                                            collate_fn = self.cfg.dss_args.collate_fn)
        
        elif self.cfg.dss_args.type == 'FacLoc':
            """
            ############################## Facility Location Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.model = model
            self.cfg.dss_args.data_type = self.cfg.dataset.type
            
            dataloader = FacLocDataLoader(trainloader, valloader, self.cfg.dss_args, logger, 
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory, 
                                          collate_fn = self.cfg.dss_args.collate_fn)
        elif self.cfg.dss_args.type == 'Full':
            """
            ############################## Full Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))

            dataloader = torch.utils.data.DataLoader(wt_trainset,
                                                     batch_size=self.cfg.dataloader.batch_size,
                                                     shuffle=self.cfg.dataloader.shuffle,
                                                     pin_memory=self.cfg.dataloader.pin_memory,
                                                     collate_fn=self.cfg.dss_args.collate_fn)
        elif self.cfg.dss_args.type == 'KAKURENBO':
            wt_trainset = WeightedSubsetX(trainset, list(range(len(trainset))), [1] * len(trainset))
            
            train_sampler = dsampler(trainset, num_replicas=1, rank=0, hidden_zero_samples=True)
            fw_sampler = dsampler(trainset, num_replicas=1, rank=0, hidden_zero_samples=False) #, shuffle=False)
            
            dataloader = torch.utils.data.DataLoader(wt_trainset,
                                                     batch_size=self.cfg.dataloader.batch_size,
                                                     sampler=train_sampler,
                                                     shuffle=False,
                                                     pin_memory=self.cfg.dataloader.pin_memory,
                                                     collate_fn=self.cfg.dss_args.collate_fn)
            dataloader_fw = torch.utils.data.DataLoader(wt_trainset,
                                                     batch_size=self.cfg.dataloader.batch_size,
                                                     sampler=fw_sampler,
                                                     shuffle=False,
                                                     pin_memory=self.cfg.dataloader.pin_memory,
                                                     collate_fn=self.cfg.dss_args.collate_fn)
            # #NguyenTT: Todo - custom sampler here                           
            # trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, sampler=train_sampler,
                                                      # pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)
            # dataloader_fw = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, sampler=log_sampler,
                                                      # pin_memory=True, collate_fn = collate_fn, drop_last=drop_last)
            # wt_trainset = trainset
            # data_loader = trainloader
        elif self.cfg.dss_args.type in ['SELCON']:
            """
            ############################## SELCON Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.lr = self.cfg.optimizer.lr
            self.cfg.dss_args.loss = criterion_nored # doubt: or criterion
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.optimizer = optimizer
            self.cfg.dss_args.criterion = criterion
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.batch_size = self.cfg.dataloader.batch_size
            
            # todo: not done yet
            self.cfg.dss_args.delta = torch.tensor(self.cfg.dss_args.delta)
            # self.cfg.dss_args.linear_layer = self.cfg.dss_args.linear_layer # already there, check glister init
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            
            dataloader = SELCONDataLoader(trainset, validset, trainloader, valloader, self.cfg.dss_args, logger,
                                           batch_size=self.cfg.dataloader.batch_size,
                                           shuffle=self.cfg.dataloader.shuffle,
                                           pin_memory=self.cfg.dataloader.pin_memory)

        else:
            raise NotImplementedError

        if self.cfg.dss_args.type in ['SELCON']:        
            is_selcon = True
        else:
            is_selcon = False


        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.cfg.ckpt.is_load:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckpt(checkpoint_path, model, optimizer)
            logger.info("Loading saved checkpoint model at epoch: {0:d}".format(start_epoch))
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                    best_acc = load_metrics['best_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics['subtrn_acc']
                if arg == "time":
                    timing = load_metrics['time']
        else:
            start_epoch = 0

        #NguyenTT
        samples_impt = torch.ones(len(trainset),dtype=torch.float64)
        samples_fness = torch.zeros(len(trainset),dtype=torch.float64) #all predict not true...
        samples_prob = torch.zeros(len(trainset),dtype=torch.float64)
        
        """
        ################################################# Training Loop #################################################
        """
        train_time = 0
        for epoch in range(start_epoch, self.cfg.train_args.num_epochs+1):
            """
            ################################################# Evaluation Loop #################################################
            """
            print_args = self.cfg.train_args.print_args
            if (epoch % self.cfg.train_args.print_every == 0) or (epoch == self.cfg.train_args.num_epochs) or (epoch == 0):
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                tst_correct = 0
                tst_total = 0
                tst_loss = 0
                model.eval()
                logger_dict = {}
                if ("trn_loss" in print_args) or ("trn_acc" in print_args):
                    samples=0
		            
                    with torch.no_grad():
                        #for _, data in enumerate(train_eval_loader):
                        for batch_idx, data in enumerate(train_eval_loader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            trn_loss += (loss.item() * trainloader.batch_size)
                            samples += targets.shape[0]
                            if "trn_acc" in print_args:
                                if is_selcon: predicted = outputs
                                else: _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()
                        trn_loss = trn_loss/samples
                        trn_losses.append(trn_loss)
                        logger_dict['trn_loss'] = trn_loss
                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)
                        logger_dict['trn_acc'] = trn_correct / trn_total

                if ("val_loss" in print_args) or ("val_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for _, data in enumerate(val_eval_loader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += (loss.item() * valloader.batch_size)
                            samples += targets.shape[0]
                            if "val_acc" in print_args:
                                if is_selcon: predicted = outputs
                                else: _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                        val_loss = val_loss/samples
                        val_losses.append(val_loss)
                        logger_dict['val_loss'] = val_loss

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)
                        logger_dict['val_acc'] = val_correct / val_total

                if ("tst_loss" in print_args) or ("tst_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for _, data in enumerate(test_eval_loader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += (loss.item() * testloader.batch_size)
                            samples += targets.shape[0]
                            if "tst_acc" in print_args:
                                if is_selcon: predicted = outputs
                                else: _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                        tst_loss = tst_loss/samples
                        tst_losses.append(tst_loss)
                        logger_dict['tst_loss'] = tst_loss

                    if (tst_correct/tst_total) > curr_best_acc:
                        curr_best_acc = (tst_correct/tst_total)

                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)
                        best_acc.append(curr_best_acc)
                        logger_dict['tst_acc'] = tst_correct / tst_total
                        logger_dict['best_acc'] = curr_best_acc

                if "subtrn_acc" in print_args:
                    if epoch == 0:
                        subtrn_acc.append(0)
                        logger_dict['subtrn_acc'] = 0
                    else:    
                        subtrn_acc.append(subtrn_correct / subtrn_total)
                        logger_dict['subtrn_acc'] = subtrn_correct / subtrn_total

                if "subtrn_losses" in print_args:
                    if epoch == 0:
                        subtrn_losses.append(0)
                        logger_dict['subtrn_loss'] = 0
                    else: 
                        subtrn_losses.append(subtrn_loss)
                        logger_dict['subtrn_loss'] = subtrn_loss

                print_str = "Epoch: " + str(epoch)
                logger_dict['Epoch'] = epoch
                logger_dict['Time'] = train_time
                timing.append(train_time)
                
                if self.cfg.train_args.wandb:
                    wandb.log(logger_dict)

                """
                ################################################# Results Printing #################################################
                """

                for arg in print_args:
                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])

                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])

                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])
                        print_str += " , " + "Best Accuracy: " + str(best_acc[-1])

                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])

                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])

                # report metric to ray for hyperparameter optimization
                if 'report_tune' in self.cfg and self.cfg.report_tune and len(dataloader) and epoch > 0:
                    tune.report(mean_accuracy=np.array(val_acc).max())

                logger.info(print_str)

            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()

            ## CUT-OFF 
            number_sample_cut = 0
            max_fraction = self.cfg.dss_args.fraction ## Need set the configuration to 0.3
            warmup_epochs = 5
            if epoch < warmup_epochs:
                new_fraction = 1*max_fraction
            elif epoch < 90:
                new_fraction = 1*max_fraction
            elif epoch < 180:
                new_fraction = 0.8*max_fraction
            elif epoch < 270:
                new_fraction = 0.6*max_fraction
            else:
                new_fraction = 0.4*max_fraction
            max_fraction = new_fraction
            MAX_sample_cut = int(max_fraction *len(samples_impt))
            ## Cut by ranking 
            sorted_impt, sorted_indices = torch.sort(samples_impt)
            number_sample_move_back = 0
            
            ## Select samples to MOVE-BACK
            THRESHOLD = 0.7
            is_samppling = torch.ones(len(trainset),dtype=torch.float64) 
            for i in range(0,MAX_sample_cut):    
                s_idx = sorted_indices[i]
                if samples_fness[s_idx] == 0 or samples_fness[s_idx] is False:
                    number_sample_move_back += 1
                    #if rank == 0 and epoch < 10:
                    #    print(str(epoch) + "\t" + str(i) + "\t" + str(s_idx) + "\t" + str(sorted_impt[i].item()), file=sample_loss_before_ex_file)
                else:
                    if samples_prob[s_idx] >= THRESHOLD:
                        number_sample_cut += 1
                        #samples_impt[s_idx] = 0 #Set weight to -1 for actual cut.
                        ### In some case, the loss of some sample is alway zero. So even it is in move-back list. it is still hided.
                        ### So we set another type of sampling weight.
                        is_samppling[s_idx] = 0 #Set weight to -1 for actual cut. 
                    else:
                        number_sample_move_back += 1
            
            train_sampler.set_epoch(epoch,is_samppling) ## Set the epoch in sampler and #Create a new indices list
            fw_sampler.set_epoch(epoch,is_samppling) ## Set the epoch in sampler and #Create a new indices list
            samples_impt = torch.zeros(len(trainset),dtype=torch.float64) #Set back to zero to fix bug of Allreduce when a rank do not calculate this samples --> it could use the old value...
            samples_fness = torch.zeros(len(trainset),dtype=torch.float64) #Set back to zero to fix bug of Allreduce when calculate the average/sum --> it could use the old value...
            samples_prob = torch.zeros(len(trainset),dtype=torch.float64) #Set back to zero to fix bug of Allreduce when calculate the average/sum --> it could use the old value...            
            #
            fraction_size.append(train_sampler.get_fraction())
        
            logger.info("{:.10f}".format(train_sampler.get_fraction()))
            print("Fraction {:.10f}".format(fraction_size[-1]))

            for batch_idx, (inputs, targets,sample_idxes, weights) in enumerate(dataloader):
                inputs = inputs.to(self.cfg.train_args.device)
                targets = targets.to(self.cfg.train_args.device, non_blocking=True)
                # weights = weights.to(self.cfg.train_args.device)
                #NguyenTT
                curent_lr = 0
                if (scheduler is not None):
                    curent_lr = scheduler.get_last_lr()
                    if epoch >= warmup_epochs:
                        lr_adj = (1/(1-max_fraction))
                        for idx, param_group in enumerate(optimizer.param_groups):
                            param_group['lr'] = curent_lr[idx] * lr_adj
                            #if batch_idx ==0 and idx==0:
                            #    print(epoch, curent_lr,param_group['lr'])
                ##
                optimizer.zero_grad()
                outputs = model(inputs)
                #NguyenTT
                log_pred = outputs.max(1, keepdim=True)[1]
                probabilities = torch.nn.functional.softmax(outputs, dim=0)
                log_prob = probabilities.max(1, keepdim=True)[0]
                log_fness = log_pred.eq(targets.view_as(log_pred))
                ###
                losses = criterion_nored(outputs, targets)
                # loss = torch.dot(losses, weights / (weights.sum()))
                loss = losses.mean()
                ##NguyenTT
                for ix in range(0, len(sample_idxes)):
                    s_idx = sample_idxes[ix]
                    #samples_visited_count[s_idx] += 1
                    #samples_per_class_count[target_batch[ix]] += 1
                    samples_impt[s_idx] = losses[ix].item()
                    samples_fness[s_idx] = log_fness[ix].item()
                    samples_prob[s_idx] = log_prob[ix].item()
                ##
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                ##NguyenTT
                if (scheduler is not None):
                    for idx, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = curent_lr[idx]  
                ##
                
                if self.cfg.scheduler.type == "cosine_annealing_step":
                    scheduler.step()
                if not self.cfg.is_reg:
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            
            ###NguyenTT - Loop for log loader
            for batch_idx, (inputs, targets,sample_idxes, weights) in enumerate(dataloader_fw):
                inputs = inputs.to(self.cfg.train_args.device)
                targets = targets.to(self.cfg.train_args.device, non_blocking=True)
                weights = weights.to(self.cfg.train_args.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                #NguyenTT
                log_pred = outputs.max(1, keepdim=True)[1]
                probabilities = torch.nn.functional.softmax(outputs, dim=0)
                log_prob = probabilities.max(1, keepdim=True)[0]
                log_fness = log_pred.eq(targets.view_as(log_pred))
                ###
                losses = criterion_nored(outputs, targets)
                loss = torch.dot(losses, weights / (weights.sum()))
                ##NguyenTT
                for ix in range(0, len(sample_idxes)):
                    s_idx = sample_idxes[ix]
                    #samples_visited_count[s_idx] += 1
                    #samples_per_class_count[target_batch[ix]] += 1
                    samples_impt[s_idx] = losses[ix].item()
                    samples_fness[s_idx] = log_fness[ix].item()
                    samples_prob[s_idx] = log_prob[ix].item()
                ##
            ###
            
            epoch_time = time.time() - start_time
            if (scheduler is not None) and (self.cfg.scheduler.type != "cosine_annealing_step"):
                scheduler.step()
            # timing.append(epoch_time)
            train_time += epoch_time           
            """
            ################################################# Checkpoint Saving #################################################
            """

            if ((epoch + 1) % self.cfg.ckpt.save_every == 0) and self.cfg.ckpt.is_save:

                metric_dict = {}

                for arg in print_args:
                    if arg == "val_loss":
                        metric_dict['val_loss'] = val_losses
                    if arg == "val_acc":
                        metric_dict['val_acc'] = val_acc
                    if arg == "tst_loss":
                        metric_dict['tst_loss'] = tst_losses
                    if arg == "tst_acc":
                        metric_dict['tst_acc'] = tst_acc
                        metric_dict['best_acc'] = best_acc
                    if arg == "trn_loss":
                        metric_dict['trn_loss'] = trn_losses
                    if arg == "trn_acc":
                        metric_dict['trn_acc'] = trn_acc
                    if arg == "subtrn_loss":
                        metric_dict['subtrn_loss'] = subtrn_losses
                    if arg == "subtrn_acc":
                        metric_dict['subtrn_acc'] = subtrn_acc
                    if arg == "time":
                        metric_dict['time'] = timing

                ckpt_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, checkpoint_path)
                logger.info("Model checkpoint saved at epoch: {0:d}".format(epoch + 1))

        """
        ################################################# Results Summary #################################################
        """
        original_idxs = set([x for x in range(len(trainset))])
        encountered_idxs = []
        if self.cfg.dss_args.type != 'Full':
            for key in dataloader.selected_idxs.keys():
                encountered_idxs.extend(dataloader.selected_idxs[key])
            encountered_idxs = set(encountered_idxs)
            rem_idxs = original_idxs.difference(encountered_idxs)
            encountered_percentage = len(encountered_idxs)/len(original_idxs)

            logger.info("Selected Indices: ") 
            logger.info(dataloader.selected_idxs)
            logger.info("Percentages of data samples encountered during training: %.2f", encountered_percentage)
            logger.info("Not Selected Indices: ")
            logger.info(rem_idxs)

            if self.cfg.train_args.wandb:
                wandb.log({
                           "Data Samples Encountered(in %)": encountered_percentage
                           })
                           
        logger.info(self.cfg.dss_args.type + " Selection Run---------------------------------")
        logger.info("Final SubsetTrn: {0:f}".format(subtrn_loss))
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                logger.info("Validation Loss: %.2f , Validation Accuracy: %.2f", val_loss, val_acc[-1])
            else:
                logger.info("Validation Loss: %.2f", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                logger.info("Test Loss: %.2f, Test Accuracy: %.2f, Best Accuracy: %.2f", tst_loss, tst_acc[-1], best_acc[-1])
            else:
                logger.info("Test Data Loss: %f", tst_loss)
        logger.info('---------------------------------------------------------------------')
        logger.info(self.cfg.dss_args.type)
        logger.info('---------------------------------------------------------------------')

        """
        ################################################# Final Results Logging #################################################
        """

        if "val_acc" in print_args:
            val_str = "Validation Accuracy: "
            for val in val_acc:
                if val_str == "Validation Accuracy: ":
                    val_str = val_str + str(val)
                else:
                    val_str = val_str + " , " + str(val)
            logger.info(val_str)

        fraction_str = "Fraction: "
        for tst in fraction_size:
            if fraction_str == "Fraction: ":
                fraction_str = fraction_str + str(tst)
            else:
                fraction_str = fraction_str + " , " + str(tst)
        logger.info(fraction_str)
        if "tst_acc" in print_args:
            tst_str = "Test Accuracy: "
            for tst in tst_acc:
                if tst_str == "Test Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

            tst_str = "Best Accuracy: "
            for tst in best_acc:
                if tst_str == "Best Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

        if "time" in print_args:
            time_str = "Time: "
            for t in timing:
                if time_str == "Time: ":
                    time_str = time_str + str(t)
                else:
                    time_str = time_str + " , " + str(t)
            logger.info(time_str)

        omp_timing = np.array(timing)
        # omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        logger.info("Total time taken by %s = %.4f ", self.cfg.dss_args.type, omp_timing[-1])
        return trn_acc, val_acc, tst_acc, best_acc, omp_timing