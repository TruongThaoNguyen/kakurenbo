## Summary
This repository contains the implementation of KAKURENBO, an method hides the sampling adaptively during the training process, in Python/PyTorch. It also contains the implementation of other methods that remove samples. All the implementations edit the original code of the baseline training python scripts case by case. 

## Implemenated methods and naming convention
We implemented the following method:
 1. The basline method: We pick-up the source code from other the other source. The training scripts are named without any postfix, e.g, `pytorch_cifar100_WRN-28-10.py`, `pytorch_imagenet_resnet50.py`, and `pytorch_imagenet_resnet50_optim_v2.py`
 2. KAKURENBO: our method, The training scripts are named with the `hs_lag_v3.1` postfix, e.g, `pytorch_cifar100_WRN-28-10_hs_lag_v3.1.py`, `pytorch_imagenet_resnet50_hs_lag_v3.1.py`, and `pytorch_imagenet_resnet50_optim_v2_hs_lag_v3.1.py` 
 3. [Importance sampling without replacement](https://arxiv.org/pdf/1803.00942.pdf) with the ``is_WR` postfix. In each iteration, each sample is chosen with a probability proportional to its loss. The with-replacement strategy means that a sample may be selected several times during an epoch, and the total number of samples fed to the model is the same as the baseline implementation
 4. [Prunning using forgeting event](https://arxiv.org/pdf/1812.05159.pdf) with the `forget_original` postfix. It trains the models in 20 epoches, collect the number of forgeting events, remove the samples from the dataset based on the forgeting event number, and then trains the models from scratch on the remaining data.
 5. [Selective Backprop](https://arxiv.org/pdf/1910.00762.pdf) with the `sb` postfix. The method prioritizes samples with high loss at each iteration. It performs the forward pass on the whole dataset, but only performs backpropagation on a subset of the dataset. Samples with higher loss has a higher possibility to be selected to perform the backward pass.

## Models and datasets
In this repository, we provide the code for the following models and datasets:
 1. **ResNet-50/EfficientNet on ImageNet-1K datasets**: we use the code from [Horovod github](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py) for the baseline. The code run on multiple GPUs accross computing nodes using Horovod framework. The baseline code is in `pytorch_imagenet_resnet50.py`. The code is added the timming measure stuff. 
 2. **ResNet-50 version 2 on ImageNet-1K datasets**: We customized the code in 1 with the hyper-parameter and techniques in [Pytoch examples - how-to-train-state-of-the-art-models](https://pytorch.org/blog\\/how-to-train-state-of-the-art-models\\-using-torchvision-latest-primitives/). The baseline code is in `pytorch_imagenet_resnet50_optim_v2.py`.
 3. **WideResNet-28-10 on CIFAR-100 datasets**: The baseline code is in `pytorch_cifar100_WRN-28-10.py`.
 4. **DeepCAM on DeepCAM dataset**: the code is omitted here.
 5. **Pretrain DeiT-Tiny-224 model on Fractal-3K dataset**: We customized the code in provided in the [CVPR2023-FDSL-on-VisualAtom github](https://github.com/masora1030/CVPR2023-FDSL-on-VisualAtom). The code are in the folder `VisualAtom` folder.

## Requirements
Each baseline code require differents library. The common library used in our experiments are:
* Python 3.x (worked at 3.8.2)
* CUDA (worked at 10.2)
* CuDNN (worked at 8.0)
* NCCL (worked at 2.7)
* OpenMPI (worked at 4.1.3)

## What we customized the baseline code.
Basically, for each use case, e.g., model and dataseet, the implementation need to edit the following:
1. **Datasets**: We track the importance of each samples, e.g., by loss or fogetting event. Todo this, we customized the dataset for return the samples index. In the main script, we import and use such kind of dataset
 ```
 from utils.folder import ImageFolder as ImageFolderX
 ...
  ## Use imageFolderX to get idx of samples when loading
    train_dataset = \
        ImageFolderX(args.train_dir,
                     transform=transforms.Compose([
                         transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                     ]))
 ```
 2.  **Sampler**: We edit the sampler to sample the samples based on the importance of samples (in is_WR) or samplng the selected samples only (other method). We customized the distributed sampler to do this.
 ```
 from utils.weight_sampler import HiddenSamplerUniform as dsampler    ### for other method
 from utils.weight_sampler import ImportanceSamplerWithReplacement as dsampler  ### for importance sampling without replacement.
 ....
     train_sampler = dsampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), hidden_zero_samples=True)
 ```
 3.  **Edit the learning rate during training**: KAKURENBO adjusts the learning rate during training. How to edit it depends on the learning_rate scheduler is used. Basically, we add the `adjust_learning_rate` to edit the learning rate and `set_learning_rate` to reset it to original learning rate.
```
def adjust_learning_rate(epoch, batch_idx, lr_scheduler, fraction):
    curent_lr = lr_scheduler.get_last_lr()
    if epoch >= args.warmup_epochs:
        lr_adj = (1/(1-fraction))
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = curent_lr[idx] * lr_adj
            if batch_idx ==0 and idx==0:
                print(epoch, curent_lr,param_group['lr'])
    return curent_lr

def set_learning_rate(epoch, batch_idx, lr):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr[idx]
```
**Note**: `CVPR2023-FDSL-on-VisualAtom` use comtimized learning rate scheduler from `timm` that do not have `get_last_lr()` function. We added this function into the scheduler of `timm` 

 4.  **Second loader for calculating the importance of hidden-samples**: After each epoch of training process, we re-calculate the loss (or forgetting event) of the samples are not envolved into training process (samples are hidded). We do this by add a new loop that perform the forward pass only.
```
# The Loop for training
for batch_idx, (sample_idxes, data, target) in enumerate(train_loader):
....
# The Loop for calculating loss
 for log_batch_idx, (log_sample_idxes, log_data, log_target) in enumerate(log_loader):
....
```
The two loader share the same dataset but samples are selected by the `hidden_zero_samples` of the sampler. 
```
    log_sampler = dsampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), hidden_zero_samples=False)
  ....
  is_samppling = ... #Select which samples are trained.
  train_sampler.set_epoch(epoch,is_samppling) ## Update the sampling indices
  log_sampler.set_epoch(epoch,is_samppling) ## Update the sampling indices
      
```

## Examples of running script
