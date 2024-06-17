#!/bin/bash

# Collect Data
python -u src/data_scripts/create_dataset.py --all --api

# Run the Plots

combinations=(
    "state --type daily"
    "state --type weekly"
    "geo --type daily --geo 0"
    "geo --type daily --geo 1"
    "geo --type weekly --geo 0"
    "geo --type weekly --geo 1"
)

for combo in "${combinations[@]}"; do
    python -u src/seasonal_plots.py --plot ${combo}
done
