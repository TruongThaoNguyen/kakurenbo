# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="cifar100",
                           datadir="../data",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=100),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.05,
                             lr1=0.1,
                             lr2=0.1,
                             lr3=0.1,
                             nesterov = True,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300,
                             stepsize=20,
                             gamma=0.1),

              dss_args=dict(type="GradMatchPB-Warm",
                            fraction=0.1,
                            select_every=20,
                            lam=0,
                            selection_type='PerBatch',
                            v1=True,
                            valid=False,
                            eps=1e-100,
                            linear_layer=True,
                            kappa=0.5,
                            collate_fn = None),


              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=1,
                              run=1,
                              wandb=False,
                              results_dir='results/',
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
