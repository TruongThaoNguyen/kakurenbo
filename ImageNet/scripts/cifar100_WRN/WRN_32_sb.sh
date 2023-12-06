
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
EPOCH=200
LR="0.025"
BETA=0.43 #1.0 #0.25

DATASET="CIFAR100"
DATA_ROOT="./data/CIFAR/${DATASET}"

LOG_DIR="./logs/${DATASET}_WRN/global_sb/G${NUM_PROCS}_F${FRACTION}_E${EPOCH}_I${JOB_ID}_sb/"

cat $SGE_JOB_HOSTLIST > ${LOG_DIR}/$JOB_ID.$JOB_NAME.nodes.list

MPIOPTS="-np ${NUM_PROCS} --hostfile $SGE_JOB_HOSTLIST --oversubscribe -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0" #-x NCCL_DEBUG=INFO"
mpirun ${MPIOPTS} python3 ./pytorch_cifar100_WRN-28-10_sb.py --dataroot ${DATA_ROOT} --log-dir ${LOG_DIR} --epochs ${EPOCH} --dataset ${DATASET} --base-lr ${LR}  --beta ${BETA} #--seed ${SEED}
#cp -r ${LOG_DIR} ${HOME_DIR}
