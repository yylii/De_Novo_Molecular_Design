#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=def-hallett-ab
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-user=yanyi.li@mail.utoronto.ca --mail-type=ALL

# Load required modules
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 amber/22

# Set input/output and topology files
PRMTOP=
INPCRD=
INPUT_DIR=
OUTPUT_DIR=

# Stage 1 - Minimization (water + ions)
pmemd.cuda -O -i $INPUT_DIR/min1.in -o $OUTPUT_DIR/min1.out \
  -p $PRMTOP -c $INPCRD -r $OUTPUT_DIR/min1.rst -ref $INPCRD

# Stage 2 - Minimization (whole system)
pmemd.cuda -O -i $INPUT_DIR/min2.in -o $OUTPUT_DIR/min2.out \
  -p $PRMTOP -c $OUTPUT_DIR/min1.rst -r $OUTPUT_DIR/min2.rst -ref $OUTPUT_DIR/min1.rst

# Stage 3 - Heating
pmemd.cuda -O -i $INPUT_DIR/heat.in -o $OUTPUT_DIR/heat.out \
  -p $PRMTOP -c $OUTPUT_DIR/min2.rst -r $OUTPUT_DIR/heat.rst \
  -x $OUTPUT_DIR/heat.nc -ref $OUTPUT_DIR/min2.rst

# Stage 4 - Equilibration 1 (with restraints)
pmemd.cuda -O -i $INPUT_DIR/density.in -o $OUTPUT_DIR/density.out \
  -p $PRMTOP -c $OUTPUT_DIR/heat.rst -r $OUTPUT_DIR/density.rst \
  -x $OUTPUT_DIR/density.nc -ref $OUTPUT_DIR/heat.rst

# Stage 5 - Equilibration 2 (no restraints)
pmemd.cuda -O -i $INPUT_DIR/equil.in -o $OUTPUT_DIR/equil.out \
  -p $PRMTOP -c $OUTPUT_DIR/density.rst -r $OUTPUT_DIR/equil.rst \
  -x $OUTPUT_DIR/equil.nc

# Stage 6 - Production MD (10 ns)
pmemd.cuda -O -i $INPUT_DIR/prod.in -o $OUTPUT_DIR/prod.out \
  -p $PRMTOP -c $OUTPUT_DIR/equil.rst -r $OUTPUT_DIR/prod.rst \
  -x $OUTPUT_DIR/prod.nc
