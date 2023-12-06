""" Scheduler Factory
Modification of timm's code.

Title: pytorch-image-models
Author: Ross Wightman
Date: 2021
Availability: https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler_factory.py
"""
from .cosine_lr import CosineLRSchedulerX as CosineLRScheduler
# from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from .linear_lr import LinearLRScheduler


def create_scheduler(args, optimizer, iter_per_epoch=1):
    num_epochs = args.epochs

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        t_mul=getattr(args, 'lr_cycle_mul', 1.),
        lr_min=args.min_lr,
        decay_rate=args.decay_rate,
        warmup_lr_init=args.warmup_lr,
        warmup_t=args.warmup_epochs,
        cycle_limit=getattr(args, 'lr_cycle_limit', 1),
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=getattr(args, 'lr_noise_pct', 0.67),
        noise_std=getattr(args, 'lr_noise_std', 1.),
        noise_seed=getattr(args, 'seed', 42),
    )
    num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs

    return lr_scheduler, num_epochs
