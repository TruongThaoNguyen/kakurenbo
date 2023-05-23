## Summary
This repository contains the implementation of KAKURENBO, an method hides the sampling adaptively during the training process, in Python/PyTorch. It also contains the implementation of other methods that remove samples. All the implementations edit the original code of the baseline training python scripts case by case. Basically, for each use case, e.g., model and dataseet, the implementation need to edit the following:

⋅⋅* Unordered sub-list. 

## Implemenated methods and naming convention
We implemented the following method:
 1. The basline method: We pick-up the source code from other the other source. The training scripts are named without any postfix, e.g, `pytorch_cifar100_WRN-28-10.py`, `pytorch_imagenet_resnet50.py`, and `pytorch_imagenet_resnet50_optim_v2.py`
 2. KAKURENBO: our method, The training scripts are named with the `hs_lag_v3.1` postfix, e.g, `pytorch_cifar100_WRN-28-10_hs_lag_v3.1.py`, `pytorch_imagenet_resnet50_hs_lag_v3.1.py`, and `pytorch_imagenet_resnet50_optim_v2_hs_lag_v3.1.py` 
 3. 
