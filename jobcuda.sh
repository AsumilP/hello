#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00

./diffusion 4098 4098 2048 16
