# Cluster SLURM job status


- All the data is on the cluster and unpacked now. I wound up having to get it directly from the host website because I couldn't compress it on kay (no write privileges in `/data/`)
- I have a working slurm script now. Testing is required to determined the right resource allocations to request.
- one issues with the script right now is that the output from the python command is not included in the output files. This needs to be fixed. 
- However, the error log is more informative than running on `kay` ever was.
- the script also 

## `puddin_slurm.sh`
```
#!/bin/bash
#SBATCH --mail-user=arh234@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -J PuddinPcc                # Job name
#SBATCH -o %x%2a.out                # Name of stdout output log file (%j expands to jobID)
#SBATCH -e %x%2a.err                # Name of stderr output log file (%j expands to jobID)
#SBATCH --open-mode=append
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH -n 1                            # Total number of cores requested
#SBATCH --mem=15G                     # Total amount of (real) memory requested (per node)
#SBATCH --time 24:00:00                  # Time limit (hh:mm:ss)
# #SBATCH --partition=  # Request partition for resource allocation
#SBATCH --array 0-11
#SBATCH --get-user-env
#SBATCH --profile=task
#SBATCH --gres=gpu:1                    # Specify a list of generic consumable resources (per node)
#SBATCH --array 0-31

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate puddin

date

echo "Job $SLURM_JOB_NAME $SLURM_JOB_ID"
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
echo "python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE} -d ${OUT_DIR}"
# time python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE} -d ${OUT_DIR}
time python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE}
umask 0077
```


- started a test with 10G of memory requested, and that was definitely not enough. `29.jsonl` was killed very quickly, though the `val` and `test` datasets endured.
- I will leave the test and val sets go for now. If they finish successfully, then the slurm script can be restarted with `--array 0-29` and more memory requested.

- the `MaxRSS` and `MaxVMSize` columns in this command (alias for `sacct --format JobId,JobName,NCPUS,MaxRSS,MaxVMSize  -T`) only show up after the job has stopped running.
  

        (base) arh234@g2-login-01:/share/compling/data/puddin$ resused --job 457740
        JobID        JobName      NCPUS     MaxRSS     MaxVMSize
        ------------ ---------- ---------- ---------- ----------
        457740_29     PuddinPcc          1
        457740_29.b+      batch          1  15294128K 172323744K
        457740_29.e+     extern          1      1168K     80212K
        457740_30     PuddinPcc          1
        457740_30.b+      batch          1
        457740_30.e+     extern          1
        457740_31     PuddinPcc          1
        457740_31.b+      batch          1
        457740_31.e+     extern          1


## `PuddinPcc29.out`

    Wed Mar 23 21:36:09 EDT 2022
    Job PuddinPcc 457741
    running on:
    - partition: default_partition
    - node(s): nikola-compute-02
    - 29 of 31
    SEED: 29 
    Task ID 29 corresponds to 'train' dataset
    Input data sourced from /share/compling/data/pile/train/29.jsonl
    =============================
    -rw-rw-r-- 1 43G Dec 31 1969 "/share/compling/data/pile/train/29.jsonl"
    -----------------------------
    Destination directory: /share/compling/data/puddin
    python /home/arh234/puddin/script/parse_pile.py -i /share/compling/data/pile/train/29.jsonl -d /share/compling/data/puddin


## `PuddinPcc29.err`

    2022-03-23 21:36:10 INFO: Loading these models for language: en (English):
    ========================
    | Processor | Package  |
    ------------------------
    | tokenize  | combined |
    | pos       | combined |
    | lemma     | combined |
    | depparse  | combined |
    ========================

    2022-03-23 21:36:10 INFO: Use device: gpu
    2022-03-23 21:36:10 INFO: Loading: tokenize
    2022-03-23 21:36:12 INFO: Loading: pos
    2022-03-23 21:36:12 INFO: Loading: lemma
    2022-03-23 21:36:12 INFO: Loading: depparse
    2022-03-23 21:36:13 INFO: Done loading processors!

    
    /var/spool/slurmd/job457741/slurm_script: line 77: 653933 Killed                  python /home/arh234/puddin/script/parse_pile.py -i ${IN_FILE}

    real	10m17.274s
    user	5m33.449s
    sys	1m1.322s
    slurmstepd: error: Detected 1 oom-kill event(s) in StepId=457741.batch cgroup. Some of your processes may have been killed by the cgroup out-of-memory handler.

## Current queue status

```
(base) arh234@g2-login-01:/share/compling/data/puddin$ sq
JOBID    NAME         ST   USER       QOS                  ACCOUNT              GROUP      PARTITION            PRIORITY   NODES TIME_LIMIT  TIME_LEFT   NODELIST(REASON)
457740_3 PuddinPcc    R    arh234     normal               compling             pug-arh234 default_partition    19334      1     1-00:00:00  23:16:24    nikola-compute-02
457740_3 PuddinPcc    R    arh234     normal               compling             pug-arh234 default_partition    19334      1     1-00:00:00  23:16:24    nikola-compute-02
```
