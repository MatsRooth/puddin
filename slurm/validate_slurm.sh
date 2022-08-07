#!/bin/bash
##SBATCH --mail-user=arh234@cornell.edu
##SBATCH --mail-type=ALL
#SBATCH -J ValidatePuddin_test-val         # Job name
#SBATCH -o %x_%j.out              # Name of stdout output log file (%j expands to jobID)
#SBATCH -e %x_%j.err              # Name of stderr output log file (%j expands to jobID)
#SBATCH --nodes=1                       # Total number of nodes requested
#SBATCH --ntasks=8                      # Total number of tasks (defaults to 1 cpu/task, but overrride with -c)
##SBATCH --cpus-per-task=8               # number of cpus per task
#SBATCH --mem-per-cpu=80G               # Total amount of (real) memory requested (per node)
#SBATCH --time 12:00:00                  # Time limit (hh:mm:ss)
#SBATCH --get-user-env
#SBATCH --chdir=/share/compling/projects/puddin/logs      # change working directory to this before execution

eval "$(conda shell.bash hook)"
conda activate puddin

echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "started @ $(date '+%F %X') from $(pwd)"
echo ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
echo "running on ${SLURMD_NODENAME} with:"
echo "  - ${SLURM_NTASKS} cores"
echo "  - ${SLURM_MEM_PER_CPU} mem/cpu"
echo "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"

DATA_DIR=/share/compling/data/puddin
python /share/compling/projects/puddin/script/confirm_doc_ids.py -d ${DATA_DIR} -g 'test' -g 'val' #-l info #>${SLURM_JOB_NAME}_${SLURM_JOB_ID}python.log
#1>logs/validate00-01.out 2>logs/validate00-01.err

echo -e "Job Closed\n$(date)"