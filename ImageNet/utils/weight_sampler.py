import math
from typing import TypeVar, Optional, Iterator
import time
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)


class ImportanceSamplerWithReplacement(Sampler[T_co]):
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
                 rank: Optional[int] = None, shuffle: bool = True,
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
              
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        
        if rank == 0:
            print("ImportanceSamplerWithReplacement", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            indices = torch.multinomial(self.weights, len(self.dataset), replacement=True, generator=g).tolist()
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
        
        stop = time.time()
        print("ImportanceSamplerWithReplacement {:.10f}".format(stop - start)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, losses) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        sum_losses = torch.sum(losses)
        self.weights = losses / sum_losses
     

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
        #Calculate fraction and threshold
        mean_weight = torch.mean(self.weights)
        std_weight = torch.std(self.weights)
        print("S:",mean_weight, std_weight)
        threshold = 0
        if std_weight > 0:
            threshold = mean_weight - std_weight #cut 20~30%
        bool_indexes = self.weights > threshold
        #indexes = bool_indexes.nonzero().data.squeeze().view(-1)  #Remove the least sample first.
        indexes = torch.nonzero(bool_indexes).data.squeeze().view(-1)  #Remove the least sample first.
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
        
class DistributedWeightedSamplerUniform(Sampler[T_co]):
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
        
        bool_indexes = torch.nonzero(self.weights).data.squeeze().view(-1)
        self.weights[bool_indexes] = 1 #set non-zero to 1 for uniform
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
  
class HiddenSamplerUniform(Sampler[T_co]):
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
                 rank: Optional[int] = None, hidden_zero_samples: bool = True, shuffle: bool = True,
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
        self.hidden_zero_samples = hidden_zero_samples
        
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
        
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        self.temp_num_samples = self.num_samples
        
        if rank == 0:
            print("HiddenSamplerUniform", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #calculate the fraction
        if self.hidden_zero_samples:
            ## Sampling from non-zero samples
            sampled_indices = torch.nonzero(self.weights).data.squeeze().view(-1)
        else:  
            ## Sampling from zero-weight samples
            bool_indexes = (self.weights == 0)
            sampled_indices = torch.nonzero(bool_indexes).data.squeeze().view(-1)
        
        if len(sampled_indices) > 0 :
            if self.drop_last and len(sampled_indices) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.temp_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(sampled_indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.temp_num_samples = math.ceil(len(sampled_indices) / self.num_replicas)  # type: ignore
            self.total_size = self.temp_num_samples * self.num_replicas 

            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                # Uniform random shuffle
                indices = torch.randperm(len(sampled_indices), generator=g).tolist()  # type: ignore
                #indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
            else:
                indices = list(range(len(sampled_indices)))  # type: ignore
                #indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
            
            if not self.drop_last:
                # add extra samples to make it evenly divisible (add item from the tail)
                if self.total_size > len(indices):
                    indices += indices[(len(indices) - self.total_size):]
                while len(indices) < self.total_size:  ## When the number of samples is too small < number of replicas
                    indices += indices[(len(indices) - self.total_size):]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:self.total_size]
                #while len(indices) < self.total_size:  ## When the number of samples is too small < number of replicas
                #    indices = indices[:self.total_size]
            
            #assert len(indices) == self.total_size
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            print(len(indices), self.temp_num_samples) 
            assert len(indices) == self.temp_num_samples
            
            self.fraction = 1 - len(indices)/self.num_samples
            
            # Use the true indices
            indices = sampled_indices[indices].tolist()
            #print(self.rank,len(sampled_indices),len(indices),max(indices),min(indices))  
        else:
            #TODO: if cutdown 100% - what to do?
            self.fraction = 0
            self.temp_num_samples = 0
            indices = []


        stop = time.time()
        print("HiddenSamplerUniform SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
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
        
    def set_hidden_mode(self, hidden_zero_samples: bool = True) -> None:    
        self.hidden_zero_samples = hidden_zero_samples

class HiddenSamplerUniform_Forget(Sampler[T_co]):
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
                 rank: Optional[int] = None, hidden_zero_samples: bool = True, shuffle: bool = True,
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
        self.hidden_zero_samples = hidden_zero_samples
        
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
        
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        self.temp_num_samples = self.num_samples
        
        if rank == 0:
            print("HiddenSamplerUniform", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #calculate the fraction
        if self.hidden_zero_samples:
            ## Sampling from non-zero samples
            sampled_indices = torch.nonzero(self.weights).data.squeeze().view(-1)
        else:  
            ## Sampling from zero-weight samples
            bool_indexes = (self.weights == 0)
            sampled_indices = torch.nonzero(bool_indexes).data.squeeze().view(-1)
        #print("AA", len(sampled_indices), max(sampled_indices), min(sampled_indices))    
        if len(sampled_indices) > 0:
            if self.drop_last and len(sampled_indices) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.temp_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(sampled_indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.temp_num_samples = math.ceil(len(sampled_indices) / self.num_replicas)  # type: ignore
            self.total_size = self.temp_num_samples * self.num_replicas 

            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                # Uniform random shuffle
                indices = torch.randperm(len(sampled_indices), generator=g).tolist()  # type: ignore
                #indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
            else:
                indices = list(range(len(sampled_indices)))  # type: ignore
                #indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
            
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
            #print(len(indices), self.temp_num_samples) 
            assert len(indices) == self.temp_num_samples
            
            self.fraction = 1 - len(indices)/self.num_samples
            
            # Use the true indices
            indices = sampled_indices[indices].tolist()
            print(self.rank,len(sampled_indices),len(indices),max(indices),min(indices))  
        else:
            #TODO: if cutdown 100% - what to do?
            self.fraction = 0
            self.temp_num_samples = 0
            indices = []


        stop = time.time()
        print("HiddenSamplerUniform SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
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
        assert len(self.weights) == len(self.dataset)
        
    def get_fraction(self):
        return self.fraction
        
    def set_hidden_mode(self, hidden_zero_samples: bool = True) -> None:    
        self.hidden_zero_samples = hidden_zero_samples
        
class HiddenSamplerUniform2(Sampler[T_co]):
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
                 rank: Optional[int] = None, hidden_zero_samples: bool = True, shuffle: bool = True,
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
        self.hidden_zero_samples = hidden_zero_samples
        
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
        
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        self.temp_num_samples = self.num_samples
        
        if rank == 0:
            print("HiddenSamplerUniform", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #calculate the fraction
        if self.hidden_zero_samples:
            ## Sampling from non-zero samples
            sampled_indices = torch.nonzero(self.weights).data.squeeze().view(-1)
            bool_indexes = (self.weights == 0)
            hide_indices = torch.nonzero(bool_indexes).data.squeeze().view(-1)
        else:  
            ## Sampling from zero-weight samples
            bool_indexes = (self.weights == 0)
            sampled_indices = torch.nonzero(bool_indexes).data.squeeze().view(-1)
            hide_indices = torch.nonzero(self.weights).data.squeeze().view(-1)
            
        if len(sampled_indices) > 0:
            if self.drop_last and len(sampled_indices) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.temp_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(sampled_indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.temp_num_samples = math.ceil(len(sampled_indices) / self.num_replicas)  # type: ignore
            self.total_size = self.temp_num_samples * self.num_replicas 

        
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                # Uniform random shuffle
                indices = torch.randperm(len(sampled_indices), generator=g).tolist()  # type: ignore
                #indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
            else:
                indices = list(range(len(sampled_indices)))  # type: ignore
                #indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
            
            
            
            ### Add the non-importance samples
            # Use the true indices
            indices = sampled_indices[indices]
            indices = torch.cat((indices, hide_indices), dim=0)
            print(self.rank, len(indices))
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
            assert len(indices) == self.temp_num_samples
            
            self.fraction = 1 - len(indices)/self.num_samples
        else:
            #TODO: if cutdown 100% - what to do?
            self.fraction = 0
            self.temp_num_samples = 0
            indices = []
            

        stop = time.time()
        print("HiddenSamplerUniform SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
        #return math.ceil(self.num_samples*(1-self.fraction))
        #return self.temp_num_samples ## should return num_samples...

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

    def get_train_length(self):
        return self.temp_num_samples ## should return num_samples...
    
    def set_hidden_mode(self, hidden_zero_samples: bool = True) -> None:    
        self.hidden_zero_samples = hidden_zero_samples
    
class HiddenSamplerUniform3(Sampler[T_co]):
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
                 rank: Optional[int] = None, hidden_zero_samples: bool = True, shuffle: bool = True,
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
        self.hidden_zero_samples = hidden_zero_samples
        
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
        
        self.fraction = 0        
        #self.num_samples = math.ceil(self.num_samples*(1-self.fraction))  #Length after remove item
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.ones(len(dataset),dtype=torch.float64)
        self.shuffle = shuffle
        self.seed = seed
        self.train_num_samples = self.num_samples
        self.hide_num_samples = 0
        self.train_total_size = self.total_size
        self.hide_total_size = 0
        
        if rank == 0:
            print("HiddenSamplerUniform", self.num_samples)

    def __iter__(self) -> Iterator[T_co]:
        start = time.time()
        #calculate the fraction
        if self.hidden_zero_samples:
            ## Sampling from non-zero samples
            sampled_indices = torch.nonzero(self.weights).data.squeeze().view(-1)
            bool_indexes = (self.weights == 0)
            hide_sampled_indices = torch.nonzero(bool_indexes).data.squeeze().view(-1)
        else:  
            ## Sampling from zero-weight samples
            bool_indexes = (self.weights == 0)
            sampled_indices = torch.nonzero(bool_indexes).data.squeeze().view(-1)
            hide_sampled_indices = torch.nonzero(self.weights).data.squeeze().view(-1)
            
        if len(sampled_indices) > 0:
            if self.drop_last and len(sampled_indices) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.train_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(sampled_indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.train_num_samples = math.ceil(len(sampled_indices) / self.num_replicas)  # type: ignore
            self.train_total_size = self.train_num_samples * self.num_replicas 
            
        
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                # Uniform random shuffle
                train_indices = torch.randperm(len(sampled_indices), generator=g).tolist()  # type: ignore
                #indices = torch.multinomial(self.weights, len(self.dataset), replacement=False, generator=g).tolist()
                
            else:
                train_indices = list(range(len(sampled_indices)))  # type: ignore
                #indices = torch.topk(self.weights, len(self.dataset)).indices.tolist()
            
            # print(len(indices),self.drop_last,self.total_size)
            if not self.drop_last:
                # add extra samples to make it evenly divisible (add item from the tail)
                if self.train_total_size > len(train_indices):
                    train_indices += train_indices[(len(train_indices) - self.train_total_size):]
            else:
                # remove tail of data to make it evenly divisible.
                train_indices = train_indices[:self.train_total_size]
            
            #print(len(indices), self.total_size)
            #assert len(indices) == self.total_size
            # subsample
            train_indices = train_indices[self.rank:self.train_total_size:self.num_replicas]
            assert len(train_indices) == self.train_num_samples
            
        else:
            #TODO: if cutdown 100% - what to do?
            self.fraction = 0
            self.train_num_samples = 0
            train_indices = []
            
        # Use the true indices
        train_indices = sampled_indices[train_indices]
        
        #### ADD HIDE_INDICE
        if len(hide_sampled_indices) > 0:
            if self.drop_last and len(hide_sampled_indices) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.hide_num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(hide_sampled_indices) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.hide_num_samples = math.ceil(len(hide_sampled_indices) / self.num_replicas)  # type: ignore
            self.hide_total_size = self.hide_num_samples * self.num_replicas 
            
            hide_indices = list(range(len(hide_sampled_indices)))  # type: ignore
            # print(len(indices),self.drop_last,self.total_size)

            if not self.drop_last:
                # add extra samples to make it evenly divisible (add item from the tail)
                if self.hide_total_size > len(hide_indices):
                    hide_indices += hide_indices[(len(hide_indices) - self.hide_total_size):]
            else:
                # remove tail of data to make it evenly divisible.
                hide_indices = hide_indices[:self.hide_total_size]
            #print(len(indices), self.total_size)
            #assert len(indices) == self.total_size
            # subsample
            hide_indices = hide_indices[self.rank:self.hide_total_size:self.num_replicas]
            assert len(hide_indices) == self.hide_num_samples
            
        else:
            self.hide_num_samples = 0
            hide_indices = []
            
        # Use the true indices
        hide_indices = hide_sampled_indices[hide_indices]
        indices = torch.cat((train_indices, hide_indices), dim=0)
        self.num_samples = self.train_num_samples + self.hide_num_samples
        self.total_size = self.train_total_size + self.hide_total_size
        self.fraction = 1 - self.train_num_samples/self.num_samples
        stop = time.time()
        ##print("AA", self.train_num_samples, self.hide_num_samples, len(hide_sampled_indices), self.num_samples, self.fraction)
        print("HiddenSamplerUniform SHUFFLING {:.10f}".format(stop - start)) if self.rank == 0 else None
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
        #return math.ceil(self.num_samples*(1-self.fraction))
        #return self.temp_num_samples

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
    
    def get_train_num_samples(self):
        return self.train_num_samples
    
    def set_hidden_mode(self, hidden_zero_samples: bool = True) -> None:    
        self.hidden_zero_samples = hidden_zero_samples
    
  