#! /bin/bash
#SBATCH --job-name=cleaned_img
#SBATCH --output=/data/project/BEACONB/logs/%j_cleaning_images.log
#SBATCH --export=none
#SBATCH --cpus-per-task=6
#SBATCH --array=1-92%6
#SBATCH --mem=6G

source /software/system/modules/latest/init/bash
module use /software/system/modules/NaN/generic
module purge
module load nan

# Load in script dependent modules here
module load miniconda/3
conda activate neuroimaging

INDEX=/data/project/BEACONB/resting/rsfMRI/.repeat_fmri
PARTICIPANT="`awk FNR==$SLURM_ARRAY_TASK_ID $INDEX`"
echo "Running on $HOSTNAME"
echo $PARTICIPANT
python /data/project/BEACONB/resting/rsfMRI/preprocessing_pipelines/grid_clean_script.py -s $PARTICIPANT