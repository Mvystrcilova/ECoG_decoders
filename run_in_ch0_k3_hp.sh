#!/bin/bash
#SBATCH --partition=volta-lp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1-00


set -eux

ch-run imgdir/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 high_pass_training.py --kernel_size 3 3 3 3 --variable 0 --starting_patient_index 1
EOF