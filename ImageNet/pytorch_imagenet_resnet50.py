import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
#from utils.image_folder_sc import ImageFolder as ImageFolderX
#from utils.local_sampler import LocalSampler as lsampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
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
### PARAMETER FOR TESTING THE IMPORTANCE SAMPLE
parser.add_argument('--log-batch-threshold', type=int, default=1,
                    help='input maximum batch idx to compute the importance of all the samples')
parser.add_argument('--log-batch-size', type=int, default=32,
                    help='input batch size for loss computation')
               


def train_old(epoch, log_dir):
    torch.cuda.synchronize()
    start_epoch = time.time()
    
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    rank = hvd.rank()
    
    if rank == 0:
        init_time_file = open(os.path.join(log_dir, "init_time.log"), "a", buffering=1)
        io_time_file = open(os.path.join(log_dir, "io_time.log"), "a", buffering=1)
        stagging_file = open(os.path.join(log_dir, "stagging.log"), "a", buffering=1)
        forward_time_file = open(os.path.join(log_dir, "forward_time.log"), "a", buffering=1)
        backward_time_file = open(os.path.join(log_dir, "backward_time.log"), "a", buffering=1)
        weightupdate_time_file = open(os.path.join(log_dir, "weightupdate_time.log"), "a", buffering=1)
        accuracy_file = open(os.path.join(log_dir, "accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "loss_per_epoch.log"), "a", buffering=1)
        accuracy_comp_file = open(os.path.join(log_dir, "accuracy_comp_iter.log"), "a", buffering=1)
        #accuracy_iter_file = open(os.path.join(log_dir, "accuracy_per_iter.log"), "a", buffering=1)
        #loss_iter_file = open(os.path.join(log_dir, "loss_per_iter.log"), "a", buffering=1)
        epoch_time_file = open(os.path.join(log_dir, "epoch_time.log"), "a", buffering=1)
        

    stop = time.time()
    print("{:.10f}".format(stop - start_epoch), file=init_time_file) if rank == 0 else None
    
    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        torch.cuda.synchronize()
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), "\t", hvd.rank(), file=io_time_file) if rank == 0 else None
            print("{:.10f}".format(stop - start), file=io_time_file) if rank == 0 else None
            
            torch.cuda.synchronize()
            start = time.time()
            adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            stop = time.time()
            print("{:.10f}".format(stop - start), file=stagging_file) if rank == 0 else None
             
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                # torch.cuda.synchronize()
                start = time.time()
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                loss = F.cross_entropy(output, target_batch)
                # torch.cuda.synchronize()
                stop = time.time()
                print("{:.10f}".format(stop - start), file=forward_time_file) if rank == 0 else None
                
                torch.cuda.synchronize()
                start = time.time()
                accuracy_iter = accuracy(output, target_batch)
                train_accuracy.update(accuracy_iter)
                train_loss.update(loss)
                # if rank == 0 and (batch_idx % number_iter_track == 0):
                    # print("{:.10f}".format(train_accuracy.avg), file=accuracy_iter_file) 
                    # print("{:.10f}".format(train_loss.avg), file=loss_iter_file) 
                # else:
                    # None
                torch.cuda.synchronize()
                stop = time.time()
                print("{:.10f}".format(stop - start), file=accuracy_comp_file) if rank == 0 else None
                
                # Average gradients among sub-batches
                # torch.cuda.synchronize()
                start = time.time()
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
                # torch.cuda.synchronize()
                stop = time.time()
                print("{:.10f}".format(stop - start), file=backward_time_file) if rank == 0 else None
                
            # Gradient is applied across all ranks
            torch.cuda.synchronize()
            start = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            stop = time.time()
            print("{:.10f}".format(stop - start), file=weightupdate_time_file) if rank == 0 else None
            
            start = time.time()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)
            torch.cuda.synchronize()
            stop = time.time()
            print("SYNC\t{:.10f}".format(stop - start), file=accuracy_comp_file) if rank == 0 else None
            
            start = time.time()

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        
    if rank == 0 :
        print("{:.10f}".format(train_accuracy.avg), file=accuracy_file) 
        print("{:.10f}".format(train_loss.avg), file=loss_file) 
    else:
        None
    
    torch.cuda.synchronize()
    stop_epoch = time.time()
    print("{:.10f}".format(stop_epoch - start_epoch), file=epoch_time_file) if rank == 0 else None
    
    if rank == 0:
        init_time_file.close()
        io_time_file.close()
        stagging_file.close()
        forward_time_file.close()
        backward_time_file.close()
        weightupdate_time_file.close()
        accuracy_file.close()
        loss_file.close()
        accuracy_comp_file.close()
        #accuracy_iter_file.close()
        #loss_iter_file.close()
        epoch_time_file.close()


def train(epoch, log_dir):
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
    if (rank is not 0):
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
    print("HAHA")    
    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)

    print("HIHI")    
    ## For compute the importance of samples. Donot shuffle for fix the sample idx
    log_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
    log_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.log_batch_size,
        sampler=log_sampler,**kwargs) 
    #init the importance
    #all_samples_impt = [None]*len(train_dataset) ## use for temporaly store the importance of exchanged samples.
    samples_impt = [None]*len(train_dataset)
    exchange_impt = [None]*len(train_dataset) 
    print("HOHO")
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
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch,args.log_dir )
        validate(epoch, args.log_dir)
        #save_checkpoint(epoch)
        # if epoch % 10 == 0:
           # save_checkpoint(epoch)