#!/bin/bash
##SBATCH --mail-user=arh234@cornell.edu
##SBATCH --mail-type=ALL
#SBATCH -J ValidatePuddin        # Job name
#SBATCH -o %x_%j.sh.out              # Name of stdout output log file (%j expands to jobID)
#SBATCH -e %x_%j.sh.err              # Name of stderr output log file (%j expands to jobID)
#SBATCH --open-mode=append
#SBATCH --nodes=1                       # Total number of nodes requested
#SBATCH --ntasks=10                      # Total number of tasks (defaults to 1 cpu/task, but overrride with -c)
##SBATCH --cpus-per-task=               # number of cpus per task
#SBATCH --mem-per-cpu=25G               # Total amount of (real) memory requested (per node)
#SBATCH --time 30:00:00                  # Time limit (hh:mm:ss)
#SBATCH --get-user-env
#SBATCH -p compling
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
(time python /share/compling/projects/puddin/script/confirm_doc_ids.py -d ${DATA_DIR} ) 2>${SLURM_JOB_NAME}_${SLURM_JOB_ID}.py.err

echo -e "\nJob Closed\n$(date)"