#!/bin/bash
#SBATCH -J base
#SBATCH -p critical
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 01-00:00:00
#SBATCH --output=/public/home/huochf/projects/image_captioning/assignment/logs/train_base.out
#SBATCH --error=/public/home/huochf/projects/image_captioning/assignment/logs/train_base.err

cd /public/home/huochf/projects/image_captioning/assignment/
python train.py --config configs/config_base.ini
