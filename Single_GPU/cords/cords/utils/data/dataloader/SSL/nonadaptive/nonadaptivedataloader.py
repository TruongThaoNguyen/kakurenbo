from torch.utils.data import DataLoader
from ..dssdataloader import DSSDataLoader
from cords.utils.data.datasets.SSL.utils import InfiniteSampler
from cords.utils.data.data_utils import WeightedSubset


class NonAdaptiveDSSDataLoader(DSSDataLoader):
    """
    Implementation of NonAdaptiveDSSDataLoader class which serves as base class for dataloaders of other
    nonadaptive subset selection strategies for semi-supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, 
                logger, *args, **kwargs):

        """
        Constructor function
        """
        super(NonAdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                       logger, *args, **kwargs)
        # Arguments assertion check
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "num_iters" in dss_args.keys(), "'num_iters' is a compulsory argument. Include it as a key in dss_args"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initialized = False
        self.num_iters = dss_args.num_iters

    def __iter__(self):
        """
        Iter function that returns the iterator of the data subset loader.
        """
        data_sub = WeightedSubset(self.dataset, self.subset_indices, self.subset_weights)
        self.curr_loader = DataLoader(data_sub, sampler=InfiniteSampler(len(data_sub), 
                                        self.num_iters * self.loader_kwargs['batch_size']),
                                         *self.loader_args, **self.loader_kwargs)
        self.batch_wise_indices = list(self.subset_loader.batch_sampler)
        return self.curr_loader.__iter__()



