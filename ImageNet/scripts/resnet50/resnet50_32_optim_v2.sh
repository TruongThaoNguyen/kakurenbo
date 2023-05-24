source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 
#hpcx-mt/2.12
source ~/venv/pytorch2023/bin/activate
export PATH=$HOME/apps/openmpi/bin:$PATH
mpicc --version

python --version
gcc --version
mpirun --version
nvcc --version

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
#NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})
EPOCH=600
LR=0.11 #LR="0.125" # RAW: 0.5 with 8 GPUs - FIX THIS VERSION
LR_Scheduler="cosineLRScheduler"  #"cosineannealinglr"
Weight_Decay=0.00001 #0.00002
LR_Warmup_Method="linear"
Label_smoothing=0.1
Batch_Size=64
Auto_Augment="ta_wide"
Random_Erase=0.4 #0.1
TRAIN_CROP=176
VAL_RESIZE=232
Norm_weight_decay=0.0
Model_ema=1 
Model_ema_steps=32 
Model_ema_decay=0.99998
BETA=1 #0.25 #0.25
FRACTION=0.3
 

cat $SGE_JOB_HOSTLIST > ${LOG_DIR}/$JOB_ID.$JOB_NAME.nodes.list
TRAIN_DIR="./train"
VAL_DIR="./val"

MPIOPTS="-np ${NUM_PROCS} --hostfile $SGE_JOB_HOSTLIST -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 -x HOROVOD_STALL_CHECK_DISABLE=1" #-x NCCL_DEBUG=INFO"

#LOG_DIR="./logs/imagenet_resnet50/global_optim_v2/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_baseline_b${Batch_Size}/"
#LOG_DIR="./logs/imagenet_resnet50/global_optim_v2/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_F${FRACTION}_hs_lag_v3.1_b${Batch_Size}/"
#LOG_DIR="./logs/imagenet_resnet50/global_optim_v2/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_F${FRACTION}_isWR_b${Batch_Size}/"
#LOG_DIR="./logs/imagenet_resnet50/global_optim_v2/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_F${FRACTION}_forget_online_b${Batch_Size}/"
#LOG_DIR="./logs/imagenet_resnet50/global_optim_v2/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_F${FRACTION}_forget_org_b${Batch_Size}/"
LOG_DIR="./logs/imagenet_resnet50/global_optim_v2/G${NUM_PROCS}_E${EPOCH}_I${JOB_ID}_BETA${BETA}_sb_b${Batch_Size}/"

rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
#mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim_v2.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} --model-ema ${Model_ema} --model-ema-steps ${Model_ema_steps} --model-ema-decay ${Model_ema_decay} --norm-weight-decay ${Norm_weight_decay} --label-smoothing ${Label_smoothing} --batch-size ${Batch_Size}
#mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim_v2_hs_lag_v3.1.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} --model-ema ${Model_ema} --model-ema-steps ${Model_ema_steps} --model-ema-decay ${Model_ema_decay} --norm-weight-decay ${Norm_weight_decay} --label-smoothing ${Label_smoothing} --batch-size ${Batch_Size} --fraction ${FRACTION}
#mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim_v2_is_wR.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} --model-ema ${Model_ema} --model-ema-steps ${Model_ema_steps} --model-ema-decay ${Model_ema_decay} --norm-weight-decay ${Norm_weight_decay} --label-smoothing ${Label_smoothing} --batch-size ${Batch_Size} --fraction ${FRACTION}
#mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim_v2_forget_lag.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} --model-ema ${Model_ema} --model-ema-steps ${Model_ema_steps} --model-ema-decay ${Model_ema_decay} --norm-weight-decay ${Norm_weight_decay} --label-smoothing ${Label_smoothing} --batch-size ${Batch_Size} --fraction ${FRACTION}
#mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim_v2_forget_org.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} --model-ema ${Model_ema} --model-ema-steps ${Model_ema_steps} --model-ema-decay ${Model_ema_decay} --norm-weight-decay ${Norm_weight_decay} --label-smoothing ${Label_smoothing} --batch-size ${Batch_Size} --fraction ${FRACTION}
mpirun ${MPIOPTS} python3 ./pytorch_imagenet_resnet50_optim_v2_sb.py --train-dir /${TRAIN_DIR} --val-dir ${VAL_DIR} --log-dir ${LOG_DIR} --epochs ${EPOCH} --base-lr ${LR} --lr-scheduler ${LR_Scheduler} --lr-warmup-method ${LR_Warmup_Method} --auto-augment ${Auto_Augment} --wd ${Weight_Decay} --random-erase ${Random_Erase} --train-crop-size ${TRAIN_CROP} --val-resize-size ${VAL_RESIZE} --model-ema ${Model_ema} --model-ema-steps ${Model_ema_steps} --model-ema-decay ${Model_ema_decay} --norm-weight-decay ${Norm_weight_decay} --label-smoothing ${Label_smoothing} --batch-size ${Batch_Size} --beta ${BETA}
#--batch-size ${Batch_Size}  --seed ${SEED}