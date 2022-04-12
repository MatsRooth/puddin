#!/bin/bash
#SBATCH --mail-user=arh234@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -J PuddinPcc                # Job name
#SBATCH -o %A_%x_slurm.out                # Name of stdout output log file (%j expands to jobID)
#SBATCH -e %A_%x_slurm.err                # Name of stderr output log file (%j expands to jobID)
#SBATCH --open-mode=append
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH -n 1                            # Total number of cores requested
#SBATCH --mem=35G                     # Total amount of (real) memory requested (per node)
#SBATCH --time 23:59:59                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition for resource allocation
#SBATCH --get-user-env
# # SBATCH --profile=task
#SBATCH --gres=gpu:1                    # Specify a list of generic consumable resources (per node)
#SBATCH --array 0-31

date; pwd

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate puddin
echo $(conda env list)

DATA_DIR=/share/compling/data
PILE_DIR=${DATA_DIR}/pile/
echo "data dir: ${DATA_DIR}"
PUD_DIR=${DATA_DIR}/puddin
if [ ! -d "$PUD_DIR" ]; then
    mkdir $PUD_DIR
fi

LOGS_DIR=${PUD_DIR}/logs
if [ ! -d "$LOGS_DIR" ]; then
    mkdir $LOGS_DIR
fi

THIS_JOB_LOG_DIR=${LOGS_DIR}/${SLURM_ARRAY_JOB_ID}
if [ ! -d "$THIS_JOB_LOG_DIR" ]; then
    mkdir $THIS_JOB_LOG_DIR
fi

echo "Job $SLURM_JOB_NAME - $SLURM_JOB_ID"
echo "  running on:"
echo "   - partition: $SLURM_JOB_PARTITION"
echo "   - node: $SLURM_JOB_NODELIST"
echo "   - 1 of $SLURM_ARRAY_TASK_COUNT"

# echo "seeding from SLURM_ARRAY_TASK_ID"
SEED=$((SLURM_ARRAY_TASK_ID))
echo "  SEED: $SEED "
PARENT=''
if [ $SEED == 30 ]; then
    echo "  Task ID ${SEED} assigned to 'test' dataset"
    SEED="test"
elif [ $SEED == 31 ]; then
    echo "  Task ID ${SEED} assigned to 'val' dataset"
    SEED="val"
else 
    echo "  Task ID ${SEED} corresponds to 'train' dataset"
    PARENT='train/'
fi

SEEDL=${#SEED}

if [ $SEEDL -lt 2 ]; then
    SEED=0$SEED
    echo "  SEED zfilled: $SEED"
fi

# set input file generated from seed (from array)
IN_DIR=${PILE_DIR}/${PARENT}
IN_FILE=${IN_DIR}${SEED}.jsonl

echo "Processing ${IN_FILE}"
echo "============================="
echo $(ls -oGghQ $IN_FILE)

echo "> Destination directory: ${DATA_DIR}"

echo "***********************************************"
echo "python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE}" -d ${DATA_DIR}"
echo ">>>>>>>>>>"
# run script and send both stdout and stderr to log file
python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE} -d ${DATA_DIR} >>"${THIS_JOB_LOG_DIR}/${SLURM_JOB_NAME}-${SEED}.log" 2>&1