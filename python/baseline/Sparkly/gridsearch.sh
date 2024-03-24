#!/bin/bash

# Assign arguments to variables
DATASET=$1
CONFIG=$2

# Execute Python script with arguments
python3 sparkly_gridsearch.py --dataset "$DATASET" --config "$CONFIG"