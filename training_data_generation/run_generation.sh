#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 30
### Partition or queue name
#SBATCH -p conroy,itc_cluster,hernquist,shared
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'generate_training'
### output and error logs
#SBATCH -o gt_%a.out
#SBATCH -e gt_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
module load Anaconda3/2019.10
source /n/home03/stacchella/.bashrc
source activate dtect
srun -n 1 python /n/eisenstein_lab/Everyone/jades/deeptect/training_data_generation/generate_images.py \
--counter="${SLURM_ARRAY_TASK_ID}" \
--config_file="/n/eisenstein_lab/Everyone/jades/deeptect/training_data_generation/config.yml" \
--output_dir="/n/eisenstein_lab/Everyone/jades/deeptect/data/training_data_20210309/"
