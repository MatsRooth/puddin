#!/bin/bash
#SBATCH --mail-user=arh234@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -J Pcc                       # Job name
#SBATCH -o %A_%x%2a.out              # Name of stdout output log file (%j expands to jobID)
#SBATCH -e %A_%x%2a.err              # Name of stderr output log file (%j expands to jobID)
#SBATCH --open-mode=append
#SBATCH -N 1                         # Total number of nodes requested
#SBATCH -n 1                         # Total number of cores requested
#SBATCH --mem=50G                    # Total amount of (real) memory requested (per node)
#SBATCH --time 71:59:59              # Time limit (hh:mm:ss)
#SBATCH --partition=gpu              # Request partition for resource allocation
#SBATCH --get-user-env
#SBATCH --gres=gpu:1                 # Specify a list of generic consumable resources (per node)
#SBATCH --array 0-31
#SBATCH --requeue                    # job will be put back in the queue if cancelled due to pre-emption or maintenance
#SBATCH --nice                       # without value defaults to --nice=100

echo "Job Array ${SLURM_JOB_NAME} #${SLURM_JOB_ID}"
date
echo "slurm stdout & stderr saved in:"
echo "     $(pwd)"
echo ""
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate puddin
echo "$(conda env list)"
echo ""
DATA_DIR=/share/compling/data
PILE_DIR=${DATA_DIR}/pile
echo "data dir: ${DATA_DIR}"
PUD_DIR=${DATA_DIR}/puddin
if [ ! -d $PUD_DIR ]; then
    mkdir $PUD_DIR
fi

LOGS_DIR=${PUD_DIR}/logs
if [ ! -d $LOGS_DIR ]; then
    mkdir $LOGS_DIR
fi
DATE="$(date -I)"
TODAY_LOGS_DIR=${LOGS_DIR}/${DATE}
if [ ! -d $TODAY_LOGS_DIR ]; then
    mkdir $TODAY_LOGS_DIR
fi

echo "  running on:"
echo "   - partition: $SLURM_JOB_PARTITION"
echo "   - node: $SLURM_JOB_NODELIST"
echo "   - $SLURM_ARRAY_TASK_COUNT total jobs in array"
echo ""
SEED=$((SLURM_ARRAY_TASK_ID))
# set input file generated from seed (from array)
SEEDIX=$SEED
if (( $SEED > 29 )); then

    if (( $SEED == 30 )); then
        SEED="Test"
    elif (( $SEED == 31 )); then
        SEED="Val"
    fi

    IN_FILE=${PILE_DIR}/${SEED,,}.jsonl

else 
    PRE0="0${SEED}"
    SEED=${PRE0:(-2)}
    IN_FILE=${PILE_DIR}/train/${SEED}.jsonl

fi

echo "Seed ${SEEDIX} -> $(ls -Qhs ${IN_FILE})"
echo ""
# run script and send both stdout and stderr to log file
LOG_FILE=${TODAY_LOGS_DIR}/${SLURM_JOB_NAME}${SEED::2}_${SLURM_ARRAY_JOB_ID:(-4)}.log
echo "Combined python log will be written to ${LOG_FILE}"

echo "***********************************************"
echo "python /home/arh234/projects/puddin/script/parse_pile.py -i ${IN_FILE} -d ${DATA_DIR} > >(tee -i -a ${LOG_FILE}) 2>&1"
echo ">>>>>>>>>>"

python /home/arh234/projects/puddin/script/parse_pile.py -i ${IN_FILE} -d ${DATA_DIR} > >(tee -i -a ${LOG_FILE}) 2>&1