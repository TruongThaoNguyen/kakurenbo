import torch
from torch.utils.data import Sampler, Dataset
import numpy as np

import os.path
from mpi4py import MPI
import random
import math
import time

class ImportanceScheduler():
    r""" Scheduler that help to communicate the local data between subset of the dataset

    Args:
        dataset: Dataset used for scheduling
        sampler: Datasampler (use with spatial_local_sampler.SpatialLocalSampler)
        comm: mpi4py communicator
        non_blocking (bool, optional): If ``True`` (default), communicated based on the non-blocking mpi4py
    """
    def __init__(self, dataset: None, non_blocking: bool = True,
            local_batch_size = 0, fraction = 0, seed = 0):

        self.dataset = dataset
        self.non_blocking = non_blocking
        self.local_batch_size = local_batch_size
        self.fraction = fraction
        self.seed = seed
        self.comm = MPI.COMM_WORLD.Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.idx_float = 0
        self.idx = 0
        random.seed(self.seed)
        self.permutation = list(range(len(dataset)))
        #random.shuffle(self.permutation)
        self.clean_list = []

        self.cp_rng = np.random.RandomState(seed)
        self.comm_targets = list(range(self.size))
        self.cp_rng.shuffle(self.comm_targets)
        
        if self.rank == 0:
            print("PatialScheduler: total ranks: {}, local samples: {}, local_batch_size: {}, fraction: {}, non-blocking: {}".format(
				self.size,
                len(self.dataset),
                local_batch_size,
                fraction, non_blocking))
            if fraction > 0:
                print("PatialScheduler: fraction: {}, float_step: {}".format(
                    fraction,
                    float(local_batch_size) * fraction))

    def clean_local_storage(self):
        for idx in self.clean_list:
            self.dataset.delete_an_item(idx)
        self.clean_list = []

    def get_clean_list(self):
        return self.clean_list
        
    def communicate(self, index):
        send_requests = []
        recv_requests = []

        if self.fraction == 0:
            return None, None

        self.idx_float += (float(self.local_batch_size) * self.fraction)
        num_idx = math.floor(self.idx_float) - self.idx
        # if self.rank == 0:
           # print("communicate: {}".format(range(self.idx, math.floor(self.idx_float))))
        # start = time.time()
        # self.cp_rng.shuffle(self.comm_targets)
        # stop = time.time()
        # print("SHUFFLE: {:.10f}".format(stop - start)) if self.rank == 0 else None
        #i = 0
        
        for idx in range(self.idx, math.floor(self.idx_float)):
            self.comm.Barrier() ## Wait for all rank to be ready for communication to avoid the csae the message before the irecv is call.
            if idx >= len(self.permutation):
                break;
                
            # Shuffle target rank list
            self.cp_rng.shuffle(self.comm_targets)
            
            # Do not communicate with self
            #target_rank = self.comm_targets[(self.rank + i) % len(self.comm_targets)]
            target_rank = self.comm_targets[self.rank]
            #src_rank = self.comm_targets.index(self.rank)
            
            #buf = bytearray(1<<22) #Create 4MB buffer  #25: 32MB...
            #buf = bytearray(1<<28) # 256MB buffer (just in case)
            buf = np.zeros(1<<22,dtype=np.uint8)
            
            if target_rank != self.rank:
                # Send to target rank
                sample, path, class_name = self.dataset.get_raw_item(self.permutation[idx])                
                send_data = {'idx':idx, 'path':path, 'sample':sample, 'class_name':class_name}
                req = self.comm.isend(send_data, dest=target_rank, tag=idx)

                send_requests.append(req)
                self.clean_list.append(self.permutation[idx])

                # Recv from ANY
                req = self.comm.irecv(buf, source=MPI.ANY_SOURCE, tag=idx) # Received will be matching by a node which is faster and in the next iter (idx) ==> mismatch of tag
                # but we set source and also barrier...
                #req = self.comm.irecv(buf, source=src_rank, tag=MPI.ANY_TAG)  # With big fraction, there is possibility of 2 message are send to 1 dest/src.
                #req = self.comm.irecv(buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG) ## Be careful because it may make the clean list mismatch...
                recv_requests.append(req)                
                
                #if self.rank == 0:
                #print("{} --> [{}]: send {}th -> rank {}".format(src_rank, self.rank, idx, target_rank))
            else:
                #print("{} --> [{}]: KEEP {}th -> rank {}".format(src_rank, self.rank, idx, target_rank))
                a = 1  # dummy code
            
            #i += 1
            ## If Blocking - wait until finish 
            if not self.non_blocking: 
                self.synchronize(send_requests,recv_requests)
                send_requests = []
                recv_requests = []

        self.idx = math.floor(self.idx_float)

        return send_requests, recv_requests


    def synchronize(self, send_requests, recv_requests):
        if self.fraction == 0:
            return

        if recv_requests is not None and len(recv_requests) > 0:
            for req in recv_requests:
                data = req.wait()
                
                self.dataset.add_a_item(self.permutation[data['idx']], 
                                                        data['path'], 
                                                        data['class_name'], 
                                                        data['sample'])
                #if self.rank == 1:
                #print("[{}]: recv {}:{} <- rank {}".format(
                #    self.rank, self.permutation[data['idx']], data['path'], (self.rank - 1) % self.size))
                
        if send_requests is not None and len(send_requests) > 0:
            for req in send_requests:
                req.wait()
        self.comm.Barrier()  ## Should remove this barrier because set 1 outsite.
        
    def scheduling(self, epoch, samples_impt):
        if self.fraction == 0:
            return

        random.seed(self.seed + epoch)
        self.idx = 0
        self.idx_float = 0
        #if self.rank == 0:
            #print("shuffle: {}".format(self.permutation))
        
        # importance sampling: select based on the top-k element with highest importance
        k = len(self.dataset)
        
        ## TOP-K
        # self.permutation = torch.topk(samples_impt, k).indices.tolist()
        
        ## Importance Sampling with Weight..
        self.permutation = torch.multinomial(samples_impt, len(samples_impt), replacement=False)
        
        
        #print("SCHEDULING",self.permutation)
        #Random sampling random.shuffle(self.permutation)

        
class ImportancePairScheduler():
    ### Pair the communication node based on the importance level, e.g., max, avg, similarity - KLdivergence?
    ### Most importance rank is pair with less importnace one ~~> to balanced the importance after exchange...
    ### FIX this order in each epoch...
    ### TODO: we can skip some pair that have not-much different (less communication...)
    r""" Scheduler that help to communicate the local data between subset of the dataset

    Args:
        dataset: Dataset used for scheduling
        sampler: Datasampler (use with spatial_local_sampler.SpatialLocalSampler)
        comm: mpi4py communicator
        non_blocking (bool, optional): If ``True`` (default), communicated based on the non-blocking mpi4py
    """
    def __init__(self, dataset: None, non_blocking: bool = True,
            local_batch_size = 0, fraction = 0, seed = 0):

        self.dataset = dataset
        self.non_blocking = non_blocking
        self.local_batch_size = local_batch_size
        self.fraction = fraction
        self.seed = seed
        self.comm = MPI.COMM_WORLD.Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.idx_float = 0
        self.idx = 0
        random.seed(self.seed)
        self.permutation = list(range(len(dataset)))
        #random.shuffle(self.permutation)
        self.clean_list = []

        self.cp_rng = np.random.RandomState(seed)
        self.comm_targets = list(range(self.size))
        self.cp_rng.shuffle(self.comm_targets)
        
        if self.rank == 0:
            print("ImportancePairScheduler: total ranks: {}, local samples: {}, local_batch_size: {}, fraction: {}, non-blocking: {}".format(
				self.size,
                len(self.dataset),
                local_batch_size,
                fraction, non_blocking))
            if fraction > 0:
                print("ImportancePairScheduler: fraction: {}, float_step: {}".format(
                    fraction,
                    float(local_batch_size) * fraction))

    def clean_local_storage(self):
        for idx in self.clean_list:
            self.dataset.delete_an_item(idx)
        self.clean_list = []

    def get_clean_list(self):
        return self.clean_list
        
    def communicate(self, index):
        send_requests = []
        recv_requests = []

        if (self.fraction == 0) or (self.comm_targets is None):
            return None, None

        self.idx_float += (float(self.local_batch_size) * self.fraction)
        num_idx = math.floor(self.idx_float) - self.idx
        # if self.rank == 0:
           # print("communicate: {}".format(range(self.idx, math.floor(self.idx_float))))
        # start = time.time()
        # self.cp_rng.shuffle(self.comm_targets)
        # stop = time.time()
        # print("SHUFFLE: {:.10f}".format(stop - start)) if self.rank == 0 else None
        #i = 0
        
        
        ## Prepare the data to send
        if num_idx > 0:
            send_buff = [0]* num_idx
            #target_rank = self.comm_targets[(self.rank + i) % len(self.comm_targets)]
            target_rank = self.comm_targets[self.rank]
            
            self.comm.Barrier() ## Wait for all rank to be ready for communication to avoid the csae the message before the irecv is call.
            # Do not communicate with self
            if target_rank != self.rank:
                #src_rank = self.comm_targets.index(self.rank)
                for sample_idx in range(0, num_idx):
                    # Send to target rank
                    idx = sample_idx + self.idx
                    if idx >= len(self.permutation):
                        break;
                    sample, path, class_name = self.dataset.get_raw_item(self.permutation[idx])                
                    send_data = {'idx':idx, 'path':path, 'sample':sample, 'class_name':class_name}
                    send_buff[sample_idx] = send_data
                    self.clean_list.append(self.permutation[idx])
                    
                req = self.comm.isend(send_buff, dest=target_rank, tag=self.idx)    
                send_requests.append(req)
                #if self.rank == 0:
                #print("{} --> [{}]: send {}th -> rank {}".format(src_rank, self.rank, idx, target_rank)) 
            
                ## Prepare data to recv
                #buf = bytearray(1<<22) #Create 4MB buffer  #25: 32MB...
                #buf = bytearray(1<<28) # 256MB buffer (just in case)
                buf = np.zeros((1<<22)* num_idx,dtype=np.uint8)

                # Recv from ANY
                #req = self.comm.irecv(buf, source=MPI.ANY_SOURCE, tag=idx) # Received will be matching by a node which is faster and in the next iter (idx) ==> mismatch of tag
                req = self.comm.irecv(buf, source=target_rank, tag=self.idx)
                # but we set source and also barrier...
                #req = self.comm.irecv(buf, source=src_rank, tag=MPI.ANY_TAG)  # With big fraction, there is possibility of 2 message are send to 1 dest/src.
                #req = self.comm.irecv(buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG) ## Be careful because it may make the clean list mismatch...
                recv_requests.append(req)                
            else:
                print("{} --> [{}]: KEEP {}th -> rank {}".format(src_rank, self.rank, idx, target_rank))
                return None, None
                
            
            #i += 1
            ## If Blocking - wait until finish 
            if not self.non_blocking: 
                self.synchronize(send_requests,recv_requests)
                send_requests = []
                recv_requests = []

            self.idx = math.floor(self.idx_float)

            return send_requests, recv_requests
        else:
            return None, None

    def synchronize(self, send_requests, recv_requests):
        if self.fraction == 0:
            return

        if recv_requests is not None and len(recv_requests) > 0:
            for req in recv_requests:
                all_data = req.wait()
                for data in all_data:
                    self.dataset.add_a_item(self.permutation[data['idx']], 
                                                            data['path'], 
                                                            data['class_name'], 
                                                            data['sample'])
                #if self.rank == 1:
                #print("[{}]: recv {}:{} <- rank {}".format(
                #    self.rank, self.permutation[data['idx']], data['path'], (self.rank - 1) % self.size))
                
        if send_requests is not None and len(send_requests) > 0:
            for req in send_requests:
                req.wait()
        self.comm.Barrier()  ## Should remove this barrier because set 1 outsite.
        
    def scheduling(self, epoch, samples_impt):
        if self.fraction == 0:
            return

        random.seed(self.seed + epoch)
        self.idx = 0
        self.idx_float = 0
   
        rank_most_impt = torch.max(samples_impt)
        np_all_most_impt  = np.zeros(self.size, dtype='d')
        self.comm.Allgather([rank_most_impt.numpy(),  MPI.DOUBLE],[np_all_most_impt, MPI.DOUBLE])
        #all_most_impt = hvd.allgather(rank_most_impt, name="importance")
        
        print(np_all_most_impt) if self.rank == 0 else None
        assert len(np_all_most_impt) == self.size
        all_most_impt = torch.from_numpy(np_all_most_impt)
        # max_most_impt = torch.max(all_most_impt)
        # min_most_impt = torch.min(all_most_impt)
        # if min_most_impt > 0.8 * max_most_impt:
            # #No exchange
            # self.comm_targets = None
            # print("Epoch [{}]- No exchanged!".format(epoch)) if self.rank == 0 else None
        # else:
            # print("Epoch [{}]- DO exchanged!".format(epoch)) if self.rank == 0 else None
        
        ## Calculate the target for pair-wise communication
        sorted_impt, sorted_indices = torch.sort(all_most_impt)
        for i in range(0, self.size):
            source = sorted_indices[i]
            dest = sorted_indices[len(sorted_indices) - (i+1)]
            self.comm_targets[source] = dest
        #if self.rank == 0:
            #print("shuffle: {}".format(self.permutation))
        
        # importance sampling: select based on the top-k element with highest importance
        k = len(self.dataset)
        
        ## TOP-K
        # self.permutation = torch.topk(samples_impt, k).indices.tolist()
        
        ## Importance Sampling with Weight..
        self.permutation = torch.multinomial(samples_impt, len(samples_impt), replacement=False)
        

        #print("SCHEDULING",self.permutation)
        #Random sampling random.shuffle(self.permutation)
