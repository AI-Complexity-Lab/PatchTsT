# CDC-FluSight-2023
Data, modeling and submission code for CDC FluSight Challenge 2023-24

## Setup environment
Use the package manager [conda](https://docs.conda.io/en/latest/) to install required Python dependencies. Note: We used Python 3.7.

```bash
conda env create -f enviroment.yml
```

## How to run
Before running, go to the src/forecaster directory.
```bash
cd ./src/forecaster
```

First run online_training.py to generate and save model predictions. Using argument -i=<exp_id> to specify an experiment parameter file <exp_id>.yaml in setup/exp_params. The results will be saved to results/base_pred/saved_pred_<data_id>.pickle. <data_id> is usually <exp_id>, and is set in setup/exp_params/<exp_id>.yaml.
```bash
python online_training.py -i=<exp_id>
```
The parameter files are all in setup folder, including covid_mortality.yaml, exp_params/<exp_id>.yaml, and seq2seq.yaml.


Then run aci.py to produce confidence intervals. The results will be saved to results/<exp_id> as a pickle file.
```bash
python aci.py -i=<exp_id>
```


The visualization and aggregated results can be produced by running eval.py. The results will be saved to results/results_<exp_id>.
```bash
python eval.py --exp_id 1000 --viz           # visualize prediction for [test week + 1, test week + weeks_ahead]
python eval.py --exp_id 1000 --save_test     # save test results
```

## Weekly submission procedure (forecasting part)
Assume current week is 202403.
1. Make sure data/weekly/weeklydata/202403.csv is available.
2. Modify parameters. In setup/covid_mortality.yaml, change input_files: weekly_data: xxxx.csv to 202403.csv. In setup/exp_params/<exp_id>.yaml, change test_week to 202403. (Usually experiments from 1 to 12 are used)
3. Goto bin/submit_exp.py, exp_ids should be [1-12].
4. In Greatlakes, run the following:
```bash
cd bin
python submit_exp.py
```
5. Pull the changes from Greatlakes once all jobs are finished (1-2 hours expected).
6. On the local computer, run the following code to generate the plots:
```bash
cd src/forecaster
python eval_all.py
```
