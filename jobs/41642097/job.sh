#!/bin/bash
#SBATCH --job-name=job
#SBATCH --partition=trevor
#SBATCH --output=jobs/running/myjob-%j.out
#SBATCH --error=jobs/running/myjob-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:30:00
#SBATCH --mem=5GB

# Export site-packages to python path
# export LD_LIBRARY_PATH=/fred/oz149/Tyler/pyenv/trevor/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/fred/oz149/Tyler/pyenv/trevor/lib/python3.9/site-packages:$PYTHONPATH
echo $ILLUSTRISTNG_API_KEY

source /fred/oz149/Tyler/pyenv/trevor/bin/activate

echo $PYTHONPATH

# Create a directory for the job
mkdir -p jobs/$SLURM_JOB_ID

# Extract the name of the submission script
SCRIPT_NAME="${SLURM_JOB_NAME}.sh"

echo "SCRIPT_NAME: $SCRIPT_NAME"

# Copy the submission script to the job directory
cp jobs/"$SCRIPT_NAME" jobs/$SLURM_JOB_ID/

# Run the compile_sources.py script
echo "Running compile_sources.py"
PYTHON_SCRIPT='src/SourceImages/compile_sources.py'

cp "$PYTHON_SCRIPT" jobs/$SLURM_JOB_ID/
python -u "$PYTHON_SCRIPT"

# move the output files to the job directory
mv jobs/running/myjob-$SLURM_JOB_ID.out jobs/$SLURM_JOB_ID/
mv jobs/running/myjob-$SLURM_JOB_ID.err jobs/$SLURM_JOB_ID/
