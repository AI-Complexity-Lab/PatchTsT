import pandas as pd
from epiweeks import Week
import os
import argparse

def add_leading_zeros(x):
    if isinstance(x, int):
        return str(x).zfill(2)
    return x

if __name__ == '__main__':
    # If Provided results number and epiweek, then a test dataset will be generated
    # Otherwise, the standard submission dataset will be created.
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='?', type=int)
    parser.add_argument('--epiweek', nargs='?', type=int)
    args = parser.parse_args()
    
    # Get the absolute path of the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the result file
    resultfile = "test_results.csv"
    result_path = os.path.join(current_directory, '..', 'results', resultfile)
    
    if args.epiweek is not None and args.results is not None:
        result_path = os.path.join(current_directory, '..', 'results', f'results_{args.results}', resultfile)
    
    # Load the CSV file using the absolute path
    df = pd.read_csv(result_path)

    final_df = pd.DataFrame()

    ew= Week.thisweek(system="CDC") # epiweek: i.e. 202301
    if args.epiweek is not None:
        ew = args.epiweek
    curyear = int(str(ew)[:-2]) # integer value of current year
    curweek = int(str(ew)[-2:]) # integer value of current week
    week = Week(curyear, curweek)

    # Create a list of quantile columns (includes week 1 to 5)
    quantile_columns = [col for col in df.columns if 'q_' in col]

    # Iterate through each quantile column
    for quantile_column in quantile_columns:
        
        quantile_value = quantile_column.split('_')[3]
        quantile_value = str(float(quantile_value)).rstrip('0').rstrip('.')
        week_number = quantile_column.split('_')[1]
        
        # Start from horizon = 0, and to go from horizon 0 -> 3
        # To start from horizon = -1, subtract 2 from int(week_number)
        horizon = int(week_number)-1
        
        quantile_df = df[['regions', quantile_column]]
        quantile_df = quantile_df.rename(columns={quantile_column: 'value'})
        quantile_df['horizon'] = horizon
        quantile_df['output_type_id'] = quantile_value
        quantile_df['reference_date'] = week.enddate()
        if(curweek + horizon) > 52:
            curyear += 1
            curweek -= 52
        quantile_df['target_end_date'] = Week(curyear, curweek + horizon).enddate()
        quantile_df['output_type'] = "quantile"
        quantile_df['target'] = "wk inc flu hosp"
        
        quantile_df = quantile_df[[
            "reference_date", "horizon", "target", "target_end_date", "regions", "output_type", "output_type_id", "value"
        ]]
        final_df = pd.concat([final_df, quantile_df], ignore_index=True)

    #Corecting Regions to Abbreviations
    fips_path = os.path.join(current_directory, '..', 'data', 'fips_codes.csv')
    fips = pd.read_csv(fips_path, header=0)
    fips = fips[['state','state_code']]
    fullfips = pd.concat([fips, pd.DataFrame({'state': ['US'], 'state_code': ['US']})], ignore_index=True)
    state_abbreviations = dict(zip(fullfips['state'], fullfips['state_code']))
    final_df['regions'] = final_df['regions'].map(state_abbreviations)
    final_df['regions'] = final_df['regions'].apply(add_leading_zeros)
    final_df = final_df.rename(columns={"regions": "location"})

    #Sort the dataframe
    final_df = final_df.sort_values(by=['horizon', 'location', 'output_type_id'])

    #Output
    eval_directory = os.path.join(current_directory, '..', '..', 'FluSight-Pred-Eval')
    if args.epiweek is None and args.results is None:
        output_file_path = os.path.join(current_directory, '..', 'output', f'{ew}.csv')
        final_df.to_csv(output_file_path, index=False)
        if os.path.exists(eval_directory):
            output_file_path = os.path.join(
                eval_directory, 'FluSight-forecast-hub', 'model-output', 'UM-DeepOutbreak', f'{week.enddate()}{"-UM-DeepOutbreak"}.csv'
            )
        else:
            output_file_path = os.path.join(
                current_directory, '..', '..', 'FluSight-forecast-hub', 'model-output', 'UM-DeepOutbreak', f'{week.enddate()}{"-UM-DeepOutbreak"}.csv'
            )
        final_df.to_csv(output_file_path, index=False)
    else:
        # Adds a test dataset in the format of a submission file to the FluSight-Pred-Eval directory
        testdata_directory = os.path.join(eval_directory, 'testdata')
        if not os.path.exists(testdata_directory):
            os.makedirs(testdata_directory)
        output_file_path = os.path.join(
            eval_directory, 'testdata', f'{week.enddate()}.csv'
        )
        if os.path.exists(output_file_path):
            print(f"{week.enddate()}.csv for week {args.epiweek} already exists")
        else:
            final_df.to_csv(output_file_path, index=False)
