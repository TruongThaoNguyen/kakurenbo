
# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.0/12.0.0 cudnn/8.8/8.8.1 nccl/2.17/2.17.1-1 gcc/12.2.0 cmake/3.26.1
# OpenMPI build
export PATH=$HOME/apps/openmpi/bin:$PATH

wandb enabled

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=tiny
export LR=1.0e-3
export CLS=3
export EPOCHS=80
export OUT_DIR=./checkpoint/${MODEL}/fdb${CLS}k/pre_training

export NGPUS=32
export NUM_PROC=4
export LOCAL_BATCH_SIZE=16
export BATCH_SIZE=$(($NGPUS*$LOCAL_BATCH_SIZE))
export INPUT_SIZE=224
# export FR=0.20
# export BETA=0.25

Baseline wihtout cutmix-and-mixup
mpirun -machinefile $SGE_JOB_HOSTLIST -npernode $NUM_PROC -np $NGPUS \
python pretrain_kakurenbo.py ./datasets/FractalDB-3000_PATCHGRAY \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_fdb${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}__80ps_noCut-mixup_OMPI \
    --input-size 3 ${INPUT_SIZE} ${INPUT_SIZE} \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5  --color-jitter 0.4 \
    --hflip 0.5 --vflip 0.5 --scale 0.08 1.0 --ratio 0.75 1.3333 \
    --epochs ${EPOCHS} --opt adamw --lr ${LR} --weight-decay 0.05 --deit-scale \
    --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 \
    --aa rand-m9-mstd0.5-inc1  --interpolation bicubic \
    --reprob 0.5 --remode pixel \
    --batch-size ${LOCAL_BATCH_SIZE} -j 18 --pin-mem \
    --drop-path 0.1 \
    --num-classes ${CLS}000 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-dir ./logs/OMPI \
    --amp \
    --log-wandb \