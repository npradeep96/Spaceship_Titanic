#!/bin/bash
#SBATCH -J MachineLearning
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc,sched_mit_arupc_long
#SBATCH -t 18:00:00
#SBATCH --mem-per-cpu 4000
cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
    source activate IDRBlocks
    python $model_script --t $train_file --v $validation_file
    conda deactivate
}

run_program

