#!/bin/bash
#SBATCH --mail-user=arh234@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -J PuddinPcc                # Job name
#SBATCH -o %x%2a.out                # Name of stdout output log file (%j expands to jobID)
#SBATCH -e %x%2a.err                # Name of stderr output log file (%j expands to jobID)
#SBATCH --open-mode=append
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH -n 1                            # Total number of cores requested
#SBATCH --mem=60G                     # Total amount of (real) memory requested (per node)
#SBATCH --time 8:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition for resource allocation
#SBATCH --get-user-env
#SBATCH --profile=task
#SBATCH --gres=gpu:1                    # Specify a list of generic consumable resources (per node)
#SBATCH --array 0-31


# activate conda environment
eval "$(conda shell.bash hook)"
conda activate puddin

date

echo "Job $SLURM_JOB_NAME, ID $SLURM_JOB_ID"
echo "  running on:"
echo "   - partition: $SLURM_JOB_PARTITION"
echo "   - node(s): $SLURM_JOB_NODELIST"
echo "   - $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX"

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

# echo 'length of seed is:' 
SEEDL=${#SEED}
# echo $SEEDL

if [ $SEEDL -lt 2 ]; then
    SEED=0$SEED
    echo "  SEED zfilled: $SEED"
fi

DATA_DIR=/share/compling/data
IN_DIR=${DATA_DIR}/pile/${PARENT}
IN_FILE=${IN_DIR}${SEED}.jsonl
# echo $IN_DIR
echo "Input data sourced from ${IN_FILE}"
echo "============================="
echo $(ls -oGghQ $IN_FILE)
echo "-----------------------------"
# echo "input line 0:"
# echo $(head -1 $IN_FILE)
# echo "_____________________________"

OUT_DIR=${DATA_DIR}/puddin
echo "Destination directory: ${OUT_DIR}"
cd $OUT_DIR

echo "***********************************************"
echo "python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE}" -d ${OUT_DIR}"
echo ">>>>>>>>>>"
time python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE} -d ${OUT_DIR}
# time python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE}

sacct -T --format=JobID,Jobname,state,start,end,elapsed,MaxRss,MaxVMSize,MaxPages,TotalCPU,NodeList -j ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
umask 0077
