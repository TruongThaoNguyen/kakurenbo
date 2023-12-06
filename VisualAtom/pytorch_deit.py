import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import zipfile
import os.path
import math
import time
from tqdm import tqdm

#### EDGAR
import yaml
from timm.models import resume_checkpoint, model_parameters, create_model, safe_model_name
from timm.utils import *
import shutil

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
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

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

### EDGAR ######## models
parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: deit_tiny_patch16_224)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')

def train(epoch, log_dir):
    rank = hvd.rank()
    start = time.time()
    
    #file_idx = int(math.floor(epoch / 10))
    #sample_loss_before_ex_filename = "BEX_loss_per_epoch_" +  str(file_idx) +".log"
                   
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
    
    model.train()
    train_sampler.set_epoch(epoch) ## Set the epoch in sampler and #Create a new indices list
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    stop = time.time()
    #print("[{}]\t{}\t{:.10f}".format(rank, epoch, stop - start), file=init_time_file) if rank == 0 else None
    
    ##### START EPOCH ######
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    start_epoch = time.time()
    #torch.cuda.synchronize()
    
    # print(rank,"START", epoch)
    send_requests, recv_requests = None, None
    with tqdm(total=len(train_loader),
             desc='Train Epoch     #{}'.format(epoch + 1),
             disable=not verbose) as t:
        #torch.cuda.synchronize()
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
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
    
    
        ## Calculate the importance
    start = time.time()
    #print(len(train_dataset), len(samples_impt)) if rank == 0 else None
        
    stop = time.time()
    impt_comp_time.update(stop-start)   
    
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    stop_epoch = time.time()
    #print("[{}]\t{}\t{:.10f}".format(rank, epoch,stop_epoch - start_epoch), file=epoch_time_file) if rank == 0 else None

    ##### END EPOCH ######
    ### REPORT TOTAL/AVG TIME
    MPI.COMM_WORLD.Barrier()
    if ((rank == 0) and (epoch ==0)):
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("RANK", "#EPOCH", "EPOCH","IO","STAG", "FW","BW","WU","ACC","LOG","IMPT"), file=report_time_file) if rank == 0 else None
    print("[{}]\t{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(rank, epoch,
        stop_epoch - start_epoch,
        io_time.sum,
        stagging_time.sum,
        forward_time.sum,
        backward_time.sum,
        wu_time.sum,
        accuracy_comp_time.sum,
        log_time.sum,
        impt_comp_time.sum
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
            for data, target in val_loader:
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
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
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

############## EDGAR def Parser Args
def _parse_args():
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    ############# Edgar Parse Args
    args, args_text = _parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    rank = hvd.rank()
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

    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    ## Import MPI here because it should call after the multiprocessing.set_start_method() call
    ## due to using of 'fork server': https://github.com/chainer/chainermn/issues/204
    from mpi4py import MPI
    #print("HAHA")    
    # val_dataset = \
    #     datasets.ImageFolder(args.val_dir,
    #                          transform=transforms.Compose([
    #                              transforms.Resize(256),
    #                              transforms.CenterCrop(224),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                   std=[0.229, 0.224, 0.225])
    #                          ]))
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
    #                                          sampler=val_sampler, **kwargs)

    #print("HIHI")    
    ## For compute the importance of samples. Donot shuffle for fix the sample idx
    # log_sampler = torch.utils.data.distributed.DistributedSampler(
        # train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
    # log_loader = torch.utils.data.DataLoader(
        # train_dataset, batch_size=args.log_batch_size,
        # sampler=log_sampler,**kwargs) 
    #init the importance
    #all_samples_impt = [None]*len(train_dataset) ## use for temporaly store the importance of exchanged samples.
    #samples_impt = [None]*len(train_dataset)
    #exchange_impt = [None]*len(train_dataset) 
    #print("HOHO")
    
    # Set up standard ResNet-50 model.
    # model = models.resnet50()
    ########### Edgar ############# model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block)
    
    if rank == 0:
        print(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
        # mkdir output dir
        os.makedirs(args.output, exist_ok=True)

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
    
    MPI.COMM_WORLD.Barrier()
    
    #### EDGAR ########### Saver
    loss_scaler = None
    eval_metric = args.eval_metric
    
    if rank == 0:
        if args.experiment:
            exp_name = args.experiment
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch,args.log_dir )
        if rank == 0:
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=eval_metric)
            if args.output:
                checkpoint_file = f'{args.output}/{args.experiment}/checkpoint-{epoch}.pth.tar'
                target_file = f'{args.output}/{args.experiment}/held-checkpoint-{epoch}.pth.tar'
            if os.path.exists(checkpoint_file):
                shutil.copyfile(checkpoint_file, target_file)
        # validate(epoch, args.log_dir)
        #save_checkpoint(epoch)
        # if epoch % 10 == 0:
           # save_checkpoint(epoch)