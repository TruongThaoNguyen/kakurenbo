import math
from typing import TypeVar, Optional, Iterator
import time
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)


class DistributedWeightedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = LocalSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, fraction = 0, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
                 
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        
        self.fraction = fraction        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        
        if rank == 0:
            print("WEIGHTED DISTRIBUTED SAMPLER", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
        else:
            #indices = list(range(len(self.dataset)))  # type: ignore
            indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
        
        # print(len(indices),self.drop_last,self.total_size)
        if not self.drop_last:
            # add extra samples to make it evenly divisible (add item from the tail)
            if self.total_size > len(indices):
                indices += indices[(len(indices) - self.total_size):]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        
        #print(len(indices), self.total_size)
        #assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        # Cut-down the tail by fraction
        real_number_samples = math.ceil(self.num_samples*(1-self.fraction))
        indices = indices[:real_number_samples]
        
        stop = time.time()
        print("WEIGHTED DISTRIBUTED SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        #return self.num_samples
        return math.ceil(self.num_samples*(1-self.fraction))

    def set_epoch(self, epoch: int, weights) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.weights = weights
            
class DistributedWeightedSamplerClip(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = LocalSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, fraction = 0, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
                 
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        
        self.fraction = fraction        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        
        if rank == 0:
            print("WEIGHTED DISTRIBUTED SAMPLER", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
        else:
            #indices = list(range(len(self.dataset)))  # type: ignore
            indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
        
        # print(len(indices),self.drop_last,self.total_size)
        if not self.drop_last:
            # add extra samples to make it evenly divisible (add item from the tail)
            if self.total_size > len(indices):
                indices += indices[(len(indices) - self.total_size):]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        
        #print(len(indices), self.total_size)
        #assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        # Cut-down the tail by fraction
        real_number_samples = math.ceil(self.num_samples*(1-self.fraction))
        indices = indices[:real_number_samples]
        
        stop = time.time()
        print("WEIGHTED DISTRIBUTED SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        #return self.num_samples
        return math.ceil(self.num_samples*(1-self.fraction))

    def set_epoch(self, epoch: int, weights, fraction) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.weights = weights
        self.fraction = fraction
        
class DistributedWeightedSamplerGauss_bug(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = LocalSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, fraction = 0, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
                 
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        
        self.factor = fraction
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.threshold = 0
        self.shuffle = shuffle
        self.seed = seed
        
        if rank == 0:
            print("WEIGHTED DISTRIBUTED SAMPLER", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #Calculate fraction and threshold
        mean_weight = torch.mean(self.weights)
        median_weight = torch.median(self.weights)
        std_weight = torch.std(self.weights)
        min_weight = torch.min(self.weights)
        #self.threshold = 0
        if std_weight > 0:
            self.threshold = mean_weight - self.factor * std_weight #cut 20~30%
        bool_indexes = self.weights > self.threshold
        print("S:",mean_weight, std_weight,median_weight, min_weight, self.threshold)
        #indexes = bool_indexes.nonzero().data.squeeze().view(-1)  #Remove the least sample first.
        indexes = torch.nonzero(bool_indexes).data.squeeze().view(-1)  #Remove the least sample first.
        if len(indexes) >= len(self.dataset):
            print("S2: logarithmic distribution")
            #historgram_impt = torch.histc(self.weights,20)
        
        self.fraction = 1 - len(indexes)/len(self.dataset)
        
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            indices = torch.multinomial(self.weights[indexes], len(indexes), replacement=False, generator=g).tolist()
        else:
            #indices = list(range(len(self.dataset)))  # type: ignore
            indices = torch.topk(self.weights, len(indexes)).indices.tolist()
        
        # print(len(indices),self.drop_last,self.total_size)
        if not self.drop_last:
            # add extra samples to make it evenly divisible (add item from the tail)
            if self.total_size > len(indices):
                indices += indices[(len(indices) - self.total_size):]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        
        #print(len(indices), self.total_size)
        #assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        # # Cut-down the tail by fraction
        # real_number_samples = math.ceil(self.num_samples*(1-self.fraction))
        # indices = indices[:real_number_samples]
        
        # Use the true indices
        indices = indexes[indices]
        
        stop = time.time()
        print("WEIGHTED DISTRIBUTED SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        #return self.num_samples
        return math.ceil(self.num_samples*(1-self.fraction))

    def set_epoch(self, epoch: int, weights) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.weights = weights
        
    def get_fraction(self):
        return self.fraction
        

class DistributedWeightedSamplerGauss2(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = LocalSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, fraction = 0, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
                 
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        
        self.factor = fraction
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.threshold = 0
        self.shuffle = shuffle
        self.seed = seed
        self.temp_num_samples = self.num_samples
        
        if rank == 0:
            print("WEIGHTED DISTRIBUTED SAMPLER", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #Calculate fraction and threshold
        mean_weight = torch.mean(self.weights)
        median_weight = torch.median(self.weights)
        std_weight = torch.std(self.weights)
        min_weight = torch.min(self.weights)
        #self.threshold = 0
        threshold = mean_weight - self.factor * std_weight #cut 20~30%
        if threshold > 0:
            self.threshold = threshold
            
        bool_indexes = self.weights > self.threshold
        print("S:",mean_weight, std_weight,median_weight, min_weight, self.threshold) if self.rank == 0 else None
        #indexes = bool_indexes.nonzero().data.squeeze().view(-1)  #Remove the least sample first.
        indexes = torch.nonzero(bool_indexes).data.squeeze().view(-1)  #Remove the least sample first.
        
        if len(indexes) > 0:
            #if len(indexes) >= len(self.dataset):
            #    print("S2: logarithmic distribution")
                #historgram_impt = torch.histc(self.weights,20)
            
            #self.fraction = 1 - len(indexes)/len(self.dataset)
            
            if self.drop_last and len(indexes) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.temp_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(indexes) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.temp_num_samples = math.ceil(len(indexes) / self.num_replicas)  # type: ignore
            total_size = self.temp_num_samples * self.num_replicas 
            
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
                indices = torch.multinomial(self.weights[indexes], len(indexes), replacement=False, generator=g).tolist()
                #indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
            else:
                #indices = list(range(len(self.dataset)))  # type: ignore
                indices = torch.topk(self.weights, len(indexes)).indices.tolist()
                #indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
            
            # print(len(indices),self.drop_last,self.total_size)
            if not self.drop_last:
                # add extra samples to make it evenly divisible (add item from the tail)
                if total_size > len(indices):
                    indices += indices[(len(indices) - total_size):]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:total_size]
            
            # subsample
            indices = indices[self.rank:total_size:self.num_replicas]
            assert len(indices) == self.temp_num_samples
            
            # # Cut-down the tail by fraction
            # real_number_samples = math.ceil(self.num_samples*(1-self.fraction))
            # indices = indices[:real_number_samples]
            
            # Use the true indices
            indices = indexes[indices]
            
            self.fraction = 1 - len(indices)/self.num_samples
        else:
            #TODO: if cutdown 100% - what to do?
            self.fraction = 0
            self.threshold = 0
            self.temp_num_samples = self.num_samples
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            else:
                indices = list(range(len(self.dataset)))  # type: ignore
            
        stop = time.time()
        print("WEIGHTED DISTRIBUTED SHUFFLING {:.10f}, Fraction {:.10f}".format(stop - start, self.fraction)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        #return self.num_samples
        #return math.ceil(self.num_samples*(1-self.fraction))
        return self.temp_num_samples

    def set_epoch(self, epoch: int, weights) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.weights = weights
        
    def get_fraction(self):
        return self.fraction

        

class DistributedWeightedSamplerGauss(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = LocalSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, fraction = 0, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
                 
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        
        self.factor = fraction
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.threshold = 0
        self.shuffle = shuffle
        self.seed = seed
        self.temp_num_samples = self.num_samples
        
        if rank == 0:
            print("WEIGHTED DISTRIBUTED SAMPLER", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #Calculate fraction and threshold
        mean_weight = torch.mean(self.weights)
        var_weight = torch.var(self.weights)
        median_weight = torch.median(self.weights)
        std_weight = torch.std(self.weights)
        min_weight = torch.min(self.weights)
        #self.threshold = 0
        threshold = mean_weight - self.factor * std_weight #cut 20~30%
        
        factor = self.factor
        factor_decay = 0.9
        while threshold < 0:
            factor = factor * factor_decay
            threshold = mean_weight - factor * std_weight #cut 20~30%
        
        self.threshold = threshold
            
        bool_indexes = self.weights > self.threshold
        print("S:",mean_weight, std_weight,median_weight, var_weight, self.threshold) if self.rank == 0 else None
        #indexes = bool_indexes.nonzero().data.squeeze().view(-1)  #Remove the least sample first.
        indexes = torch.nonzero(bool_indexes).data.squeeze().view(-1)  #Remove the least sample first.
        
        if len(indexes) > 0:
            #if len(indexes) >= len(self.dataset):
            #    print("S2: logarithmic distribution")
                #historgram_impt = torch.histc(self.weights,20)
            
            #self.fraction = 1 - len(indexes)/len(self.dataset)
            
            if self.drop_last and len(indexes) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.temp_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(indexes) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.temp_num_samples = math.ceil(len(indexes) / self.num_replicas)  # type: ignore
            total_size = self.temp_num_samples * self.num_replicas 
            
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
                indices = torch.multinomial(self.weights[indexes], len(indexes), replacement=False, generator=g).tolist()
                #indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
            else:
                #indices = list(range(len(self.dataset)))  # type: ignore
                indices = torch.topk(self.weights, len(indexes)).indices.tolist()
                #indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
            
            # print(len(indices),self.drop_last,self.total_size)
            if not self.drop_last:
                # add extra samples to make it evenly divisible (add item from the tail)
                if total_size > len(indices):
                    indices += indices[(len(indices) - total_size):]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:total_size]
            
            # subsample
            indices = indices[self.rank:total_size:self.num_replicas]
            assert len(indices) == self.temp_num_samples
            
            # # Cut-down the tail by fraction
            # real_number_samples = math.ceil(self.num_samples*(1-self.fraction))
            # indices = indices[:real_number_samples]
            
            # Use the true indices
            indices = indexes[indices]
            
            self.fraction = 1 - len(indices)/self.num_samples
        else:
            #TODO: if cutdown 100% - what to do?
            self.fraction = 0
            self.threshold = 0
            self.temp_num_samples = self.num_samples
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            else:
                indices = list(range(len(self.dataset)))  # type: ignore
            
        stop = time.time()
        print("WEIGHTED DISTRIBUTED SHUFFLING {:.10f}, Fraction {:.10f}".format(stop - start, self.fraction)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        #return self.num_samples
        #return math.ceil(self.num_samples*(1-self.fraction))
        return self.temp_num_samples

    def set_epoch(self, epoch: int, weights) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.weights = weights
        
    def get_fraction(self):
        return self.fraction
 