#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --output=jobs/running/myjob-%j.out
#SBATCH --error=jobs/running/myjob-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=10GB

# Actiate the virtual environment
PYENV_NAME=deeplenstronomy

# Export site-packages to python path
export LD_LIBRARY_PATH=/fred/oz149/Tyler/pyenv/$PYENV_NAME/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/fred/oz149/Tyler/pyenv/$PYENV_NAME/lib/python3.7/site-packages:${PYTHONPATH}
source /fred/oz149/Tyler/pyenv/$PYENV_NAME/bin/activate
echo Python Path: $PYTHONPATH

# If the job is part of an array then create a subdirectory for the array job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p jobs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/$SLURM_ARRAY_TASK_ID
    OUT_DIR=jobs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/$SLURM_ARRAY_TASK_ID
    JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
fi

# If the job is not part of an array create the job directory
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p jobs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    OUT_DIR=jobs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
fi

# Extract the name of the submission script
SCRIPT_NAME="${SLURM_JOB_NAME}.sh"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Copy the submission script to the job directory
cp jobs/"$SCRIPT_NAME" $OUT_DIR

# Run the python script
echo "Running data_preprocessing.py"
PYTHON_SCRIPT='src/data_preprocessing.py'
MODEL='epl'

cp "$PYTHON_SCRIPT" $OUT_DIR
python -u "$PYTHON_SCRIPT" $MODEL

# move the output files to the job directory
mv jobs/running/myjob-$SLURM_JOB_ID.out $OUT_DIR
mv jobs/running/myjob-$SLURM_JOB_ID.err $OUT_DIR