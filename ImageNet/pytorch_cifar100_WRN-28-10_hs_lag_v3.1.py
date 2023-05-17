import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from utils.folder import ImageFolder as ImageFolderX
from utils.weight_sampler import HiddenSamplerUniform as dsampler
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
    
    #file_idx = int(math.floor(epoch / 10))
    #sample_loss_before_ex_filename = "BEX_loss_per_epoch_" +  str(file_idx) +".log"
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
    
    ## Calculate the importance at the 1 epoch
    start = time.time()
    global samples_impt
    global samples_fness
    # global samples_pred_flip
    # global samples_pred_true_count
    global samples_prob
    #global samples_visited_count
    # if epoch %5 == 0:
        # samples_impt = torch.zeros(len(train_dataset),dtype=torch.float64)
        # for log_batch_idx, (sample_idxes, log_data, log_target) in enumerate(log_loader):  ##Loading data 4s
            # if args.cuda:           # Around 0.1s
                # log_data, log_target = log_data.cuda(), log_target.cuda()
            # log_output = model(log_data)        #Around 0.5s

            # log_loss = F.cross_entropy(log_output, log_target,reduction='none')
            # log_loss = log_loss
            
            # for i in range(0, len(sample_idxes)):
                # s_idx = sample_idxes[i]
                # samples_impt[s_idx] = log_loss[i].item()
    if epoch > 0:
        samples_impt = hvd.allreduce(samples_impt, average=False, name="importance")
        samples_fness = hvd.allreduce(samples_fness, average=False, name="fness") ##TODO: a sample which is sampled 2 times in 1 epoch in 2 different rank may have double value...
        # samples_pred_flip = hvd.allreduce(samples_pred_flip, name="fness")
        # samples_pred_flip = torch.ceil(samples_pred_flip)
        # #samples_pred_flip = math.ceil(float(samples_pred_flip)/ size)
        # samples_pred_true_count = hvd.allreduce(samples_pred_true_count, name="fness")
        # samples_pred_true_count = torch.ceil(samples_pred_true_count)
        samples_prob = hvd.allreduce(samples_prob, average=False, name="prob")
    stop = time.time()
    impt_comm_time.update(stop-start)
    
    
    ### ANALYSIS
    #historgram_impt = torch.histogram(samples_impt,20)
    ##historgram_impt = torch.histc(samples_impt,20)
    #mean_impt = torch.mean(samples_impt)
    #std_impt = torch.std(samples_impt)
    #if rank == 0:
    #    print(str(epoch) + "\tM " + str(mean_impt)  + "\tS " + str(std_impt) + "\tH " + str(historgram_impt), file=loss_histogram_file)
    ############    
    
    # std = torch.where(samples_impt <= mean_impt, mean_impt-samples_impt, torch.zeros(len(samples_impt),dtype=torch.float64))
    # indexes = torch.nonzero(std).data.squeeze().view(-1)
    # sstd = math.sqrt(torch.sum(torch.pow(std,2))/(len(indexes) -1))
    # threshold = mean_impt - sstd
    # fraction = torch.count_nonzero(samples_impt < threshold) / len(samples_impt)
    
    #fraction = args.fraction
    #max_iter_idx = math.ceil((1-fraction)*len(train_loader))  
    #sorted_impt, sorted_indices = torch.sort(samples_impt)
    
    ## CUT-OFF 
    number_sample_cut = 0
    # number_sample_cut_flip = 0
    # number_sample_cut_pred = 0
    max_fraction = adjust_fraction(epoch, args.fraction)
    MAX_sample_cut = int(max_fraction *len(samples_impt))
    ## Cut by ranking 
    sorted_impt, sorted_indices = torch.sort(samples_impt)
    number_sample_move_back = 0
    
    # ### Analysis 2
    # top5_low_loss_number_sample = math.ceil(5*len(samples_impt)/100)
    # top5_low_loss_mean = torch.mean(sorted_impt[0:top5_low_loss_number_sample])
    # top95_high_loss_mean = torch.mean(sorted_impt[top5_low_loss_number_sample:len(samples_impt)])
    
    # top95_low_loss_number_sample = math.ceil(95*len(samples_impt)/100)
    # top95_low_loss_mean = torch.mean(sorted_impt[0:top95_low_loss_number_sample])
    # top5_high_loss_mean = torch.mean(sorted_impt[top95_low_loss_number_sample:len(samples_impt)])
    # if rank == 0:
        # print(str(epoch) + "\tt5  " + str(top5_low_loss_mean)  + "\t" + str(top95_high_loss_mean) + "\tt95 " + str(top95_low_loss_mean)  + "\t" + str(top5_high_loss_mean), file=loss_histogram_file)
    # #####
    
    ## Select samples to MOVE-BACK
    THRESHOLD = 0.7
    is_samppling = torch.ones(len(train_dataset),dtype=torch.float64) 
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
                

        # if float(samples_pred_flip[s_idx])/(epoch+1) < 0.1: # 90% predict correctly so far
            # number_sample_cut_flip += 1
            
        # if samples_pred_true_count[s_idx] > 3:
            # number_sample_cut_pred += 1
        
    #number_sample_cut = number_sample_cut - number_sample_move_back
    #max_iter_idx = math.ceil((1 - number_sample_cut/len(samples_impt)) *len(train_loader))
    #fraction = (1-max_iter_idx/len(train_loader))
    print(rank, epoch, number_sample_move_back, number_sample_cut,MAX_sample_cut, len(samples_impt))
    if rank == 0:
        print("C\t" + str(epoch)  + "\t" + str(number_sample_move_back)+ "\t" + str(number_sample_cut) + "\t" + str(MAX_sample_cut) + "\t" + str(len(samples_impt)), file= loss_histogram_file)

    stop = time.time()
    impt_comp_time.update(stop-start)    

    train_sampler.set_epoch(epoch,is_samppling) ## Set the epoch in sampler and #Create a new indices list
    log_sampler.set_epoch(epoch,is_samppling) ## Set the epoch in sampler and #Create a new indices list
    samples_impt = torch.zeros(len(train_dataset),dtype=torch.float64) #Set back to zero to fix bug of Allreduce when a rank do not calculate this samples --> it could use the old value...
    samples_fness = torch.zeros(len(train_dataset),dtype=torch.float64) #Set back to zero to fix bug of Allreduce when calculate the average/sum --> it could use the old value...
    samples_prob = torch.zeros(len(train_dataset),dtype=torch.float64) #Set back to zero to fix bug of Allreduce when calculate the average/sum --> it could use the old value...
    
    if rank == 0:
        print("{:.10f}".format(train_sampler.get_fraction()), file=fraction_file)
        #print("MAX_ITER",max_iter_idx)
    #if rank == 0:
    #    print(str(epoch) + "\t" + str(mean_impt)  + "\t" + str(std_impt), file=loss_histogram_file)
    
    ##### START EPOCH ######
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    start_epoch = time.time()
    #torch.cuda.synchronize()
    # print(rank,"START", epoch)
    #number_sample_miss = 0
    #under_confidence_number = 0
    # samples_per_class_count = torch.zeros(1000, dtype=torch.long)
    # samples_per_class_cut_count = torch.zeros(1000, dtype=torch.long)
    
    with tqdm(total=len(train_loader), ## TODO: Old length
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
            adjust_learning_rate(epoch, batch_idx, max_fraction)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            #torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), file=stagging_file) if rank == 0 else None
            stagging_time.update(stop - start) 
                
            # Split data into sub-batches of size batch_size
            #if batch_idx <= max_iter_idx:
            for i in range(0, len(data), args.batch_size):
                #torch.cuda.synchronize()
                start = time.time()
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]

                output = model(data_batch)                       
                log_pred = output.max(1, keepdim=True)[1]
                probabilities = torch.nn.functional.softmax(output, dim=0)
                log_prob = probabilities.max(1, keepdim=True)[0]
                log_fness = log_pred.eq(target_batch.view_as(log_pred))
                #if batch_idx == 0:
                    # print(rank, "O ",output)
                    # print(rank, "PRED", log_pred)
                    # print(rank, "FNESS",log_fness)
                    # print(rank, "PROB", log_prob)
                #if batch_idx < 5:
                #    print(log_pred, target_batch, log_fness)
                
                #loss = F.cross_entropy(output, target_batch)
                losses = F.cross_entropy(output, target_batch, reduction='none')
                loss = losses.mean()
                #torch.cuda.synchronize()
                stop = time.time()
                #print("{:.10f}".format(stop - start), file=forward_time_file) if rank == 0 else None
                forward_time.update(stop-start)

                start = time.time()
                for ix in range(0, len(sample_idxes)):
                    s_idx = sample_idxes[ix]
                    #samples_visited_count[s_idx] += 1
                    #samples_per_class_count[target_batch[ix]] += 1
                    samples_impt[s_idx] = losses[ix].item()
                    samples_fness[s_idx] = log_fness[ix].item()
                    samples_prob[s_idx] = log_prob[ix].item()
                    
                    # if (samples_fness[s_idx] == 1 or samples_fness[s_idx] is True) and samples_prob[s_idx] < THRESHOLD:
                        # under_confidence_number += 1 
                    # if samples_fness[s_idx] == 0 or samples_fness[s_idx] is False:
                        # samples_pred_flip[s_idx] += 1
                        # samples_pred_true_count[s_idx] = 0
                    # else: # predict correctly
                        # samples_pred_true_count[s_idx] += 1

                    #if batch_idx == 0:
                    #    print(rank, s_idx, samples_fness[s_idx])
                    #if samples_fness[s_idx] > 0 and batch_idx < 1:
                    #    print("HAHA", rank, s_idx, ix, log_fness[ix], samples_fness[s_idx])
                    #    print("HAHA", rank, s_idx, ix, log_fness[ix], samples_fness[s_idx], log_prob[ix])
                    #if batch_idx < 2:
                    #    print (samples_fness[s_idx] , s_idx)
                    
                stop = time.time()
                impt_comp_time.update(stop-start)    
                
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
    with tqdm(total=len(log_loader), ## TODO: Old length
             desc='Imp_Cal Epoch     #{}'.format(epoch + 1),
             disable=not verbose) as t:
        start = time.time()
        for log_batch_idx, (log_sample_idxes, log_data, log_target) in enumerate(log_loader):
            if (log_batch_idx == 0) and (rank == 0):
                print ("Actual iter:", len(log_loader))
            stop = time.time()
            #print("{:.10f}".format(stop - start), file=io_time_file) if rank == 0 else None
            io_time.update(stop - start)
            
            #torch.cuda.synchronize()
            start = time.time()
            if args.cuda:
                log_data, log_target = log_data.cuda(), log_target.cuda()
            #torch.cuda.synchronize()
            stop = time.time()
            #print("{:.10f}".format(stop - start), file=stagging_file) if rank == 0 else None
            stagging_time.update(stop - start) 
            
            with torch.no_grad(): 
                for i in range(0, len(log_data), args.batch_size):
                    # for ix in range(0, len(sample_idxes)):
                        # s_idx = sample_idxes[ix]
                        # if samples_fness[s_idx] == 0 or samples_fness[s_idx] is False:
                            # number_sample_miss += 1
                    
                    #torch.cuda.synchronize()
                    start = time.time()
                    log_data_batch = log_data[i:i + args.batch_size]
                    log_target_batch = log_target[i:i + args.batch_size]

                    log_output = model(log_data_batch)
                    
                    log_pred = log_output.max(1, keepdim=True)[1]
                    probabilities = torch.nn.functional.softmax(log_output, dim=0)
                    log_prob = probabilities.max(1, keepdim=True)[0]
                    log_fness = log_pred.eq(log_target_batch.view_as(log_pred))

                    #loss = F.cross_entropy(log_output, log_target_batch)
                    losses = F.cross_entropy(log_output, log_target_batch, reduction='none')
                    loss = losses.mean()
                    #torch.cuda.synchronize()
                    stop = time.time()
                    #print("{:.10f}".format(stop - start), file=forward_time_file) if rank == 0 else None
                    forward_time.update(stop-start)

                    start = time.time()
                    for ix in range(0, len(log_sample_idxes)):
                        s_idx = log_sample_idxes[ix]
                        #samples_per_class_cut_count[log_target_batch[ix]] += 1
                        samples_impt[s_idx] = losses[ix].item()
                        samples_fness[s_idx] = log_fness[ix].item()
                        samples_prob[s_idx] = log_prob[ix].item()
                        
                        # if (samples_fness[s_idx] == 1 or samples_fness[s_idx] is True) and samples_prob[s_idx] < THRESHOLD:
                            # under_confidence_number += 1 
                        # if samples_fness[s_idx] == 0 or samples_fness[s_idx] is False:
                            # samples_pred_flip[s_idx] += 1
                            # samples_pred_true_count[s_idx] = 0
                        # else:
                            # samples_pred_true_count[s_idx] += 1
                            # if log_prob[ix] < 0.5:
                                # under_confidence_number += 1
                    stop = time.time()
                    impt_comp_time.update(stop-start)

                    t.set_postfix({'loss': loss}) #Local accuracy at rank 0
                    t.update(1)                    
        start = time.time()
    #if rank ==0:
    
    # samples_per_class_count = hvd.allreduce(samples_per_class_count, average=False, name="classcount")
    # samples_per_class_cut_count = hvd.allreduce(samples_per_class_cut_count, average=False, name="classcount2")
    # if rank == 0 :
        # print("SC",samples_per_class_count.tolist())
        # print("CC",samples_per_class_cut_count.tolist())  
        # print("D",rank, epoch,under_confidence_number)   
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    stop_epoch = time.time()
    #print("[{}]\t{}\t{:.10f}".format(rank, epoch,stop_epoch - start_epoch), file=epoch_time_file) if rank == 0 else None

    ##### END EPOCH ######
    ### REPORT TOTAL/AVG TIME
    if ((rank == 0) and (epoch ==0)):
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("RANK", "#EPOCH", "EPOCH","IO","STAG", "FW","BW","WU","ACC","LOG","IMPT_COMP","IMPT_COMM"), file=report_time_file) if rank == 0 else None
    MPI.COMM_WORLD.Barrier()
    # print("[{}]\t{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}".format(rank, epoch,
        # stop_epoch - start_epoch,
        # io_time.sum,
        # stagging_time.sum,
        # forward_time.sum,
        # backward_time.sum,
        # wu_time.sum,
        # accuracy_comp_time.sum,
        # log_time.sum,
        # impt_comp_time.sum,
        # impt_comm_time.sum
        # ))
    
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

    # correct_samples_per_class_count = torch.zeros(1000, dtype=torch.long)
    # total_samples_per_class_count = torch.zeros(1000, dtype=torch.long)
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for sample_idxes, data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                # log_pred = output.max(1, keepdim=True)[1]
                # probabilities = torch.nn.functional.softmax(output, dim=0)
                # log_prob = probabilities.max(1, keepdim=True)[0]
                # log_fness = log_pred.eq(target.view_as(log_pred))
                # for ix in range(0, len(sample_idxes)):
                    # total_samples_per_class_count[target[ix]] += 1
                    # s_fness = log_fness[ix].item()
                    # if (s_fness == 1 or s_fness is True):
                        # correct_samples_per_class_count[target[ix]] += 1
                
                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)
    # correct_samples_per_class_count = hvd.allreduce(correct_samples_per_class_count, average=False, name="classcount")
    # total_samples_per_class_count = hvd.allreduce(total_samples_per_class_count, average=False, name="classcount2")
    # if rank == 0 :
        # print("CRC",correct_samples_per_class_count.tolist())
        # print("TTC",total_samples_per_class_count.tolist())
        
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
def adjust_learning_rate(epoch, batch_idx, fraction):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 60:
        lr_adj = (1.0) * (1/(1-fraction))
    elif epoch < 120:
        lr_adj = (0.2) * (1/(1-fraction))
    elif epoch < 160:
        lr_adj = (0.04)* (1/(1-fraction))
    else:
        lr_adj = (0.008)* (1/(1-fraction))
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj

def adjust_fraction2(epoch,fraction):  #_decay
    max_fraction = fraction
    if epoch < args.warmup_epochs:
        new_fraction = 1*max_fraction
    elif epoch < 60:
        new_fraction = 0.8*max_fraction
    elif epoch < 120:
        new_fraction = 0.6*max_fraction
    elif epoch < 160:
        new_fraction = 0.4*max_fraction
    else:
        new_fraction = 0.2*max_fraction
    return new_fraction
    
def adjust_fraction(epoch,fraction):  #_decay
    max_fraction = fraction
    if epoch < args.warmup_epochs:
        new_fraction = 1*max_fraction
    elif epoch < 60:
        new_fraction = 1*max_fraction
    elif epoch < 120:
        new_fraction = 0.8*max_fraction
    elif epoch < 160:
        new_fraction = 0.6*max_fraction
    else:
        new_fraction = 0.4*max_fraction
    return new_fraction
    
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
    train_sampler = dsampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), hidden_zero_samples=True)
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
    log_sampler = dsampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), hidden_zero_samples=False) #, shuffle=False)
    log_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.log_batch_size,
        sampler=log_sampler,**kwargs) 
    #init the importance
    #all_samples_impt = [None]*len(train_dataset) ## use for temporaly store the importance of exchanged samples.
    samples_impt = torch.ones(len(train_dataset),dtype=torch.float64)
    samples_fness = torch.zeros(len(train_dataset),dtype=torch.float64) #all predict not true...
    # samples_pred_true_count = torch.zeros(len(train_dataset),dtype=torch.float64) 
    # samples_pred_flip = torch.zeros(len(train_dataset),dtype=torch.float64) 
    samples_prob = torch.zeros(len(train_dataset),dtype=torch.float64)
    # samples_visited_count = torch.zeros(len(train_dataset),dtype=torch.float64)

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