
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 hpcx-mt/2.12
source ~/venv/pytorch2023/bin/activate

python --version
gcc --version
mpirun --version
nvcc --version


NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
#NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})
EPOCH=600
LR="0.125" # RAW: 0.5 with 8 GPUs
LR_Scheduler="cosineannealinglr"
LR_Warmup_Method="linear"
Auto_Augment="ta_wide"
Weight_Decay=0.00002
Random_Erase=0.1
TRAIN_CROP=176
VAL_RESIZE=232
FRACTION=0.3



cat $SGE_JOB_HOSTLIST > ${LOG_DIR}/$JOB_ID.$JOB_NAME.nodes.list
TRAIN_DIR="./train"
VAL_DIR="./val"

MPIOPTS="-np ${NUM_PROCS} --hostfile $SGE_JOB_HOSTLIST -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 -x HOROVOD_STALL_CHECK_DISABLE=1" #-x NCCL_DEBUG=INFO"

LOG_DIR="./logs/imagenet_resnet50/global_optim/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_baseline_fixLR/"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} #--batch-size ${Batch_Size}  --seed ${SEED}
