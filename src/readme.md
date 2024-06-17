## Notes

## Submission
- To create the submission file, run the script from a parent directory that has CDC-FluSight-2023 and FluSight-Pred-Eval folders included. In this case, run:

```bash 
python3 CDC-FluSight-2023/src/submission.py
```

- This submission script will add a submission file in the output folder

## Plots
- To create the seasonal plots for this week, use:

```bash
chmod +x ./bin/run_plots.sh
```

```bash
./bin/run_plots.sh
```

- Regional Plots and State Plots are created:
    - State Plots will be generated with weekly and daily data points
    - Regional Plots will be generated with weekly and daiy data points
        - States are combined into regions like NorthEast, West, South, Midwest

## Evaluation
- How to evaluate the results: 
    - Download the FluSight-pred-eval repository
    - Follow the Instructions from that repository