import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from utils.lr_scheduler.scheduler_factory import create_scheduler
#from utils.image_folder_sc import ImageFolder as ImageFolderX
#from utils.local_sampler import LocalSampler as lsampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import utils.presets as presets
import utils.optim_utils as optim_utils
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
parser.add_argument('--base-lr', type=float, default=0.5,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4,
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
#### NEW SETTING
parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
parser.add_argument( "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")               
parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
#parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: constant)")
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
#### SETTING OF OPTIM-V2#####
parser.add_argument("--model-ema", type=int, default=0, help="enable tracking Exponential Moving Average of model parameters")
parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)",)
parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)
parser.add_argument("--norm-weight-decay",default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)",)
parser.add_argument("--bias-weight-decay",default=None,type=float,help="weight decay for bias parameters of all layers (default: None, same value as --wd)",)
parser.add_argument("--transformer-embedding-decay",default=None,type=float,help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",)
parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")  #Not implemented
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
            help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
                    
def train(epoch, log_dir, model_ema=None):
    rank = hvd.rank()
    start = time.time()
    
    file_idx = int(math.floor(epoch / 10))
    sample_loss_before_ex_filename = "BEX_loss_per_epoch_" +  str(file_idx) +".log"
                   
    # Init log file
    if rank ==0:
        init_time_file = open(os.path.join(log_dir, "init_time.log"), "a", buffering=1)
        io_time_file = open(os.path.join(log_dir, "io_time.log"), "a", buffering=1)
        stagging_file = open(os.path.join(log_dir, "stagging.log"), "a", buffering=1)
        forward_time_file = open(os.path.join(log_dir, "forward_time.log"), "a", buffering=1)
        backward_time_file = open(os.path.join(log_dir, "backward_time.log"), "a", buffering=1)
        weightupdate_time_file = open(os.path.join(log_dir, "weightupdate_time.log"), "a", buffering=1)
        accuracy_file = open(os.path.join(log_dir, "accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "loss_per_epoch.log"), "a", buffering=1)
        scheduler_time_file = open(os.path.join(log_dir, "scheduler.log"), "a", buffering=1)
        accuracy_comp_file = open(os.path.join(log_dir, "accuracy_comp_iter.log"), "a", buffering=1)
        #accuracy_iter_file = open(os.path.join(log_dir, "accuracy_per_iter.log"), "a", buffering=1)
        #loss_iter_file = open(os.path.join(log_dir, "loss_per_iter.log"), "a", buffering=1)
        epoch_time_file = open(os.path.join(log_dir, "epoch_time.log"), "a", buffering=1)
        report_time_file =open(os.path.join(log_dir, "report_time.log"), "a", buffering=1)
        sample_loss_before_ex_file =open(os.path.join(log_dir, sample_loss_before_ex_filename), "a", buffering=1)
    
    
    MPI.COMM_WORLD.Barrier()
    #Wait for rank 0 create first.
    if (rank != 0):
        report_time_file =open(os.path.join(log_dir, "report_time.log"), "a", buffering=1)
        sample_loss_before_ex_file =open(os.path.join(log_dir, sample_loss_before_ex_filename), "a", buffering=1)
    
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
            #adjust_learning_rate(epoch, batch_idx)
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

                # start = time.time()
                # if ((epoch % 10 == 0) and (batch_idx <= args.log_batch_threshold) and (batch_idx > 0)) or ((batch_idx == 1)):         
                    # for sample_idx, (log_data, log_target) in enumerate(log_loader):
                        # if args.cuda:
                            # log_data, log_target = log_data.cuda(), log_target.cuda()
                        # log_output = model(log_data)
                        # log_loss = F.cross_entropy(log_output, log_target)
                        # if torch.isnan(log_loss):
                            # print(log_output, log_target)
                        # #print("{}_{}\t{}\t{}\t{:.10f}".format(rank, sample_idx, epoch, batch_idx, log_loss), file=loss_per_rank_file)
                        # print("{}_{}\t{}\t{}\t{:.10f}".format(rank, sample_idx, epoch, batch_idx, log_loss), file=sample_loss_file)
                    # #importance_calculate(epoch, batch_idx, loss_per_rank_file)
                # stop = time.time()
                # accuracy_comp_time.update(stop-start)
                
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
            
            #### MODEL EMA
            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)            
            #####
            
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
    
    # for log_batch_idx, (log_data, log_target) in enumerate(log_loader):
        # # if s_idx > 10:
            # # break
        # if args.cuda:
            # log_data, log_target = log_data.cuda(), log_target.cuda()
        # #print(log_data)
        # log_output = model(log_data)
        # log_output = log_output.chunk(args.log_batch_size)
        # log_target = log_target.chunk(args.log_batch_size)
        # #print (len(log_target),len(log_output)) if rank == 0 else None
        # for s_offset in range (0, len(log_output)):
            # s_idx = s_offset + log_batch_idx * args.log_batch_size
            # log_loss = F.cross_entropy(log_output[s_offset], log_target[s_offset])
            # samples_impt[s_idx] = log_loss.item()  #.item: detach from graph                
            # if torch.isnan(log_loss):
                # print(s_idx, log_output[s_offset], log_target[s_offset])
            # print("{}_{}\t{}\t{:.10f}".format(rank, s_idx, epoch, log_loss), file=sample_loss_before_ex_file)    
        
    stop = time.time()
    impt_comp_time.update(stop-start)   
    
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    stop_epoch = time.time()
    #print("[{}]\t{}\t{:.10f}".format(rank, epoch,stop_epoch - start_epoch), file=epoch_time_file) if rank == 0 else None

    ##### END EPOCH ######
    ### REPORT TOTAL/AVG TIME
    #print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("RANK", "#EPOCH",
    #    "EPOCH","IO","STAG", "FW","BW","WU","ACC","LOG"), file=report_time_file) if rank == 0 else None
    MPI.COMM_WORLD.Barrier()
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
        init_time_file.close()
        io_time_file.close()
        stagging_file.close()
        scheduler_time_file.close()
        forward_time_file.close()
        backward_time_file.close()
        weightupdate_time_file.close()
        accuracy_file.close()
        loss_file.close()
        accuracy_comp_file.close()
        #accuracy_iter_file.close()
        #loss_iter_file.close()
        epoch_time_file.close()
        #print(rank,"END", epoch)
    report_time_file.close()
    sample_loss_before_ex_file.close()
        
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

        
    ### NEW - VERSION2
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)    
    auto_augment_policy = args.auto_augment
    random_erase_prob = args.random_erase
    
    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                             presets.ClassificationPresetTrain(
                                crop_size=train_crop_size,
                                interpolation=interpolation,
                                auto_augment_policy=auto_augment_policy,
                                random_erase_prob=random_erase_prob,),)
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
    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                             presets.ClassificationPresetEval(
                                crop_size=val_crop_size, 
                                resize_size=val_resize_size, 
                                interpolation=interpolation),
                            )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)

    #### END NEW####

    # Set up standard ResNet-50 model.
    model = models.resnet50()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    ### WEIGHT DECAY TUNNING
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = optim_utils.set_weight_decay(
        model,
        args.wd,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
            
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(parameters, #model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, 
                          weight_decay=args.wd)
    #####
    
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
    
    ### OPTIMIZER LEARNING RATE SCHEDULER
    print(args.epochs, args.warmup_epochs, args.epochs-args.warmup_epochs)
    #main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr_min
    #)
    main_lr_scheduler, num_iters = create_scheduler(args, optimizer, len(train_loader))
    
    # if args.warmup_epochs > 0:
        # if args.lr_warmup_method == "linear":
            # warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                # optimizer, start_factor=args.lr_warmup_decay, total_iters=args.warmup_epochs
            # )
        # elif args.lr_warmup_method == "constant":
            # warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                # optimizer, factor=args.lr_warmup_decay, total_iters=args.warmup_epochs
            # )
        # else:
            # raise RuntimeError(
                # f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            # )
        # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            # optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_epochs]
        # )
    # else:
    lr_scheduler = main_lr_scheduler
    ####
    
    #### MODEL EMA
    model_without_ddp = model
    ### We may not use the ddp --> comment it
    # if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model_without_ddp = model.module
    model_ema = None
    if args.model_ema > 0:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        device = hvd.local_rank()
        model_ema = optim_utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    ## SKIP the code resume with model-ema and test the model_ema
    ####
    
    MPI.COMM_WORLD.Barrier()
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch,args.log_dir,model_ema)
        lr_scheduler.step(epoch + 1)
        validate(epoch,args.log_dir)
        #save_checkpoint(epoch)
        # if epoch % 10 == 0:
           # save_checkpoint(epoch)