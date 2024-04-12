#!/bin/bash
#SBATCH --job-name=data_simulation
#SBATCH --output=jobs/running/myjob-%j.out
#SBATCH --error=jobs/running/myjob-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=5GB
#SBATCH --array=0-10

# Variables
NUMBER_OF_IMAGES=100
RESOULTION=0.08

# Actiate the virtual environment
PYENV_NAME=deeplenstronomy

# Export site-packages to python path
export LD_LIBRARY_PATH=/fred/oz149/Tyler/pyenv/$PYENV_NAME/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/fred/oz149/Tyler/pyenv/$PYENV_NAME/lib/python3.7/site-packages:${PYTHONPATH}
source /fred/oz149/Tyler/pyenv/$PYENV_NAME/bin/activate
echo $PYTHONPATH

# If the job is part of an array then create a subdirectory for the array job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p jobs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/$SLURM_ARRAY_TASK_ID
    OUT_DIR=jobs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/$SLURM_ARRAY_TASK_ID
    JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
fi

# If the job is not part of an array create the job directory
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    OUT_DIR=jobs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
fi

# Extract the name of the submission script
SCRIPT_NAME="${SLURM_JOB_NAME}.sh"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Copy the submission script to the job directory
cp jobs/"$SCRIPT_NAME" $OUT_DIR

# Copy the config files
cp -r data/config_files $OUT_DIR

# Run the compile_sources.py script
echo "Running data_simulation.py"
PYTHON_SCRIPT='src/data_simulation.py'

echo $((NUMBER_OF_IMAGES/SLURM_ARRAY_TASK_MAX)) images per job

cp "$PYTHON_SCRIPT" $OUT_DIR
python -u "$PYTHON_SCRIPT" $((NUMBER_OF_IMAGES/SLURM_ARRAY_TASK_MAX)) 0.08 $JOB_NAME $SLURM_ARRAY_TASK_ID

# move the output files to the job directory
mv jobs/running/myjob-$SLURM_JOB_ID.out $OUT_DIR
mv jobs/running/myjob-$SLURM_JOB_ID.err $OUT_DIR
