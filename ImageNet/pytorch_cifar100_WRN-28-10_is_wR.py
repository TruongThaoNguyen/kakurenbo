import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from utils.folder import ImageFolder as ImageFolderX
from utils.weight_sampler import ImportanceSamplerWithReplacement as dsampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from pytorch_cifar100.models.wideresidual import wideresnet
from utils.datasets_cifar import CIFAR100 as cifar100
import horovod.torch as hvd
import os
import zipfile
import os.path
import math
import time
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default='./', type=str)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--dataset', default='CIFAR100', type=str)
# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.0005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
### PARAMETER FOR TESTING THE IMPORTANCE SAMPLE
parser.add_argument('--log-batch-threshold', type=int, default=1,
                    help='input maximum batch idx to compute the importance of all the samples')
parser.add_argument('--log-batch-size', type=int, default=32,
                    help='input batch size for loss computation')
parser.add_argument('--fraction', type=float, default=0.0,
                    help=' in range of [0,1]. If ratio = 0, it is normal importance sampling, fraction>0, remove fractions of samples from dataset during 1 epoch. Default value = 0')

def train(epoch, log_dir):
    rank = hvd.rank()
    start = time.time()
    
    loss_histogram_filename = "loss_hist_per_epoch.log"
                   
    # Init log file
    if rank ==0:
        #init_time_file = open(os.path.join(log_dir, "init_time.log"), "a", buffering=1)
        #io_time_file = open(os.path.join(log_dir, "io_time.log"), "a", buffering=1)
        #stagging_file = open(os.path.join(log_dir, "stagging.log"), "a", buffering=1)
        #forward_time_file = open(os.path.join(log_dir, "forward_time.log"), "a", buffering=1)
        #backward_time_file = open(os.path.join(log_dir, "backward_time.log"), "a", buffering=1)
        #weightupdate_time_file = open(os.path.join(log_dir, "weightupdate_time.log"), "a", buffering=1)
        accuracy_file = open(os.path.join(log_dir, "accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "loss_per_epoch.log"), "a", buffering=1)
        #scheduler_time_file = open(os.path.join(log_dir, "scheduler.log"), "a", buffering=1)
        #accuracy_comp_file = open(os.path.join(log_dir, "accuracy_comp_iter.log"), "a", buffering=1)
        #accuracy_iter_file = open(os.path.join(log_dir, "accuracy_per_iter.log"), "a", buffering=1)
        #loss_iter_file = open(os.path.join(log_dir, "loss_per_iter.log"), "a", buffering=1)
        #epoch_time_file = open(os.path.join(log_dir, "epoch_time.log"), "a", buffering=1)
        report_time_file =open(os.path.join(log_dir, "report_time.log"), "a", buffering=1)
        #sample_loss_before_ex_file =open(os.path.join(log_dir, sample_loss_before_ex_filename), "a", buffering=1)
        loss_histogram_file =open(os.path.join(log_dir, loss_histogram_filename), "a", buffering=1)
        fraction_file = open(os.path.join(log_dir, "fraction_per_epoch.log"), "a", buffering=1)
    
    MPI.COMM_WORLD.Barrier()
    #Wait for rank 0 create first.
    if (rank != 0):
        report_time_file =open(os.path.join(log_dir, "report_time.log"), "a", buffering=1)
        #sample_loss_before_ex_file =open(os.path.join(log_dir, sample_loss_before_ex_filename), "a", buffering=1)
    
    io_time = TimeEstimation("io_time")
    stagging_time = TimeEstimation("stagging_time")
    forward_time = TimeEstimation("forward_time")
    backward_time = TimeEstimation("backward_time")
    wu_time = TimeEstimation("wu_time")
    accuracy_comp_time = TimeEstimation("accuracy_comp_time")
    epoch_time = TimeEstimation("epoch_time")
    log_time=TimeEstimation("log_time")
    impt_comp_time = TimeEstimation("importance_comp_time")
    impt_comm_time = TimeEstimation("importance_comm_time")
    
    model.train()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    stop = time.time()
    #print("[{}]\t{}\t{:.10f}".format(rank, epoch, stop - start), file=init_time_file) if rank == 0 else None
    
    ##### START EPOCH ######
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    start_epoch = time.time()
    #torch.cuda.synchronize()
    
    ## Calculate the importance
    start = time.time()
    if epoch == 0:
        samples_impt = torch.ones(len(train_dataset),dtype=torch.float64)
    else:
        samples_impt = torch.zeros(len(train_dataset),dtype=torch.float64)
        
        for log_batch_idx, (sample_idxes, log_data, log_target) in enumerate(log_loader):  ##Loading data 4s
            if args.cuda:           # Around 0.1s
                log_data, log_target = log_data.cuda(), log_target.cuda()
            #print(log_data)
            log_output = model(log_data)        #Around 0.5s
            log_loss = F.cross_entropy(log_output, log_target,reduction='none')
            log_loss = log_loss
            
            for i in range(0, len(sample_idxes)):
                s_idx = sample_idxes[i]
                samples_impt[s_idx] = log_loss[i].item()
    stop = time.time()
    impt_comp_time.update(stop-start)    

    start = time.time()
    if epoch > 0:
        samples_impt = hvd.allreduce(samples_impt, average=False, name="importance")
    stop = time.time()
    impt_comm_time.update(stop-start)

    train_sampler.set_epoch(epoch,samples_impt) ## Set the epoch in sampler and #Create a new indices list
    
    # print(rank,"START", epoch)
    with tqdm(total=len(train_loader),
             desc='Train Epoch     #{}'.format(epoch + 1),
             disable=not verbose) as t:
        #torch.cuda.synchronize()
        start = time.time()
        for batch_idx, (sample_idxes, data, target) in enumerate(train_loader):
            #torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), file=io_time_file) if rank == 0 else None
            io_time.update(stop - start)
            
            #torch.cuda.synchronize()
            start = time.time()
            adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            #torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), file=stagging_file) if rank == 0 else None
            stagging_time.update(stop - start) 
            
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                #torch.cuda.synchronize()
                start = time.time()
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                loss = F.cross_entropy(output, target_batch)
                
                #torch.cuda.synchronize()
                stop = time.time()
                #print("{:.10f}".format(stop - start), file=forward_time_file) if rank == 0 else None
                forward_time.update(stop-start)
                
                #torch.cuda.synchronize()
                #start = time.time()
                #accuracy_iter = accuracy(output, target_batch)
                #train_accuracy.update(accuracy_iter)
                # train_loss.update(loss)
                # if rank == 0 and (batch_idx % number_iter_track == 0):
                    # print("{:.10f}".format(train_accuracy.avg), file=accuracy_iter_file) 
                    # print("{:.10f}".format(train_loss.avg), file=loss_iter_file) 
                # else:
                    # None
                # #torch.cuda.synchronize()
                # stop = time.time()
                #print("{:.10f}".format(stop - start), file=accuracy_comp_file) if rank == 0 else None
                # accuracy_comp_time.update(stop-start)
                
                # Average gradients among sub-batches
                #torch.cuda.synchronize()
                start = time.time()
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
                #torch.cuda.synchronize()
                stop = time.time()
                #print("{:.10f}".format(stop - start), file=backward_time_file) if rank == 0 else None
                backward_time.update(stop-start)
            
            # Gradient is applied across all ranks
            #torch.cuda.synchronize()
            start = time.time()
            optimizer.step()
            #torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), file=weightupdate_time_file) if rank == 0 else None
            wu_time.update(stop-start)
            
            start = time.time()
            #t.set_postfix({'accuracy': 100. * train_accuracy.avg.item()})
            t.set_postfix({'loss': loss}) #Local accuracy at rank 0
            t.update(1)
            #torch.cuda.synchronize()
            stop = time.time()
            #print("SYNC\t{:.10f}".format(stop - start), file=accuracy_comp_file) if rank == 0 else None
            log_time.update(stop-start)
            
            start = time.time()
    
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    stop_epoch = time.time()
    #print("[{}]\t{}\t{:.10f}".format(rank, epoch,stop_epoch - start_epoch), file=epoch_time_file) if rank == 0 else None

    ##### END EPOCH ######
    ### REPORT TOTAL/AVG TIME
    if ((rank == 0) and (epoch ==0)):
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("RANK", "#EPOCH", "EPOCH","IO","STAG", "FW","BW","WU","ACC","LOG","IMPT_COMP","IMPT_COMM"), file=report_time_file) if rank == 0 else None
    MPI.COMM_WORLD.Barrier()
    print("[{}]\t{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(rank, epoch,
        stop_epoch - start_epoch,
        io_time.sum,
        stagging_time.sum,
        forward_time.sum,
        backward_time.sum,
        wu_time.sum,
        accuracy_comp_time.sum,
        log_time.sum,
        impt_comp_time.sum,
        impt_comm_time.sum
        ), file=report_time_file) if rank == 0 else None
    
    ### REPORT ACCURACY (using the last accuracy)
    accuracy_iter = accuracy(output, target_batch)
    train_accuracy.update(accuracy_iter)
    train_loss.update(loss)
    
    start = time.time()
    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        
    if rank == 0 :
        print("{:.10f}".format(train_accuracy.avg), file=accuracy_file) 
        print("{:.10f}".format(train_loss.avg), file=loss_file) 
    else:
        None
    stop = time.time()
    

    if rank ==0:
        #init_time_file.close()
        #io_time_file.close()
        #stagging_file.close()
        #scheduler_time_file.close()
        #forward_time_file.close()
        #backward_time_file.close()
        #weightupdate_time_file.close()
        accuracy_file.close()
        loss_file.close()
        #accuracy_comp_file.close()
        #accuracy_iter_file.close()
        #loss_iter_file.close()
        #epoch_time_file.close()
        fraction_file.close()
        loss_histogram_file.close()
        #print(rank,"END", epoch)
    report_time_file.close()
    #sample_loss_before_ex_file.close()
        
def validate(epoch, log_dir):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    rank = hvd.rank()
    
    if rank ==0:
        accuracy_file = open(os.path.join(log_dir, "val_accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "val_loss_per_epoch.log"), "a", buffering=1)

    
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for sample_idxes, data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
    if rank == 0 :
        print("{:.10f}".format(val_accuracy.avg), file=accuracy_file) 
        print("{:.10f}".format(val_loss.avg), file=loss_file) 
    else:
        None
    if rank ==0:
        accuracy_file.close()
        loss_file.close()

# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
#  On CIFAR learning rate dropped by 0.2 at 60, 120 and 160 epochsand we train for total 200 epoch
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 60:
        lr_adj = 1.
    elif epoch < 120:
        lr_adj = 0.2
    elif epoch < 160:
        lr_adj = 0.04
    else:
        lr_adj = 0.008
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        filepath = os.path.join(args.log_dir, filepath)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

        
class TimeEstimation(object):
    def __init__(self, name):
        self.name = name
        self.sum_ = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum_ += val
        self.n += 1

    @property
    def avg(self):
        return self.sum_ / self.n

    @property
    def sum(self):
        return self.sum_

if __name__ == '__main__':
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    print ("Run with arguments:")
    for key, value in args._get_kwargs():
        if value is not None:
            print(value,key) if hvd.rank() == 0 else None
    
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    num_classes = 100 if args.dataset == 'CIFAR100' else 10 ## Only support CIFAR
        
    # From the WRN paper, "In general, we observed that CIFAR mean/std preprocessing allows training wider anddeeper networks with better accuracy"
    # Preprocessing with mean/std. Value pick up from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)    
        
    if args.dataset == 'CIFAR100':    
        train_dataset = \
            cifar100(args.dataroot,
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
                                 
        val_dataset = \
            cifar100(args.dataroot,
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
    else:
        train_dataset = \
            datasets.CIFAR10(args.dataroot,
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
                                 
        val_dataset = \
            datasets.CIFAR10(args.dataroot,
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                                    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
                                 ]))
        
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    #train_sampler = dsampler(
    #    train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), hidden_zero_samples=True)
    train_sampler = dsampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    ## Import MPI here because it should call after the multiprocessing.set_start_method() call
    ## due to using of 'fork server': https://github.com/chainer/chainermn/issues/204
    from mpi4py import MPI
        
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)

    ## For compute the importance of samples. Donot shuffle for fix the sample idx
    log_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()) #, shuffle=False)
    log_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.log_batch_size,
        sampler=log_sampler,**kwargs) 
    #init the importance
    #all_samples_impt = [None]*len(train_dataset) ## use for temporaly store the importance of exchanged samples.
    samples_impt = torch.ones(len(train_dataset),dtype=torch.float64)

    # Set up Wide_RESNET-28-10 model.
    #model = Wide_ResNet(28, 10, 0, num_classes)
    model = wideresnet(28,10) 

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    

    
    ## COUNT THE NUMBER OF SAMPLE PER CLASS
    # labels = torch.zeros(1000, dtype=torch.long)
    # for index, sample, target in train_dataset:
        # labels[target] += 1
    # if hvd.rank() == 0:
        # print(labels)
        
    MPI.COMM_WORLD.Barrier()
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch,args.log_dir)
        validate(epoch, args.log_dir)
        #save_checkpoint(epoch)
        # if epoch % 10 == 0:
           # save_checkpoint(epoch)