import pandas as pd
import numpy as np
from symptoms import querydata
import os
from dateutil.relativedelta import relativedelta
from epiweeks import Week
import covidcast
import threading
from datetime import datetime
from sodapy import Socrata

"""
    Generates daily data files
"""

# Convert HHS Date Time to Standard
def convert_date_format(date_str, api):
    if api:
        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
    return date_obj.strftime('%Y-%m-%d')


def rundaily(epiweek, start_date, end_date, state_fips, states):
    """
    Step 2: Collect data from each source
    """
    covidcast.use_api_key("450127ae273dd")

    #Fetch Data
    def fetch_data(source, states, alldata, data_lock):
        data = covidcast.signal(source["data_source"], 
                                source["signal"], 
                                start_date, 
                                end_date, 
                                "state", 
                                geo_values=states)
        with data_lock:
            alldata.append(data)

    api_source = [
        {"data_source": "doctor-visits", "signal": "smoothed_adj_cli"},
        {"data_source": "hhs", "signal": "confirmed_admissions_covid_1d"},
        {"data_source": "hhs", "signal": "confirmed_admissions_influenza_1d"}
    ]

    alldata = []

    #Run Threads
    threads = []
    data_lock = threading.Lock()
    for source in api_source:
        thread = threading.Thread(target=fetch_data, args=(source, states, alldata, data_lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    #Mapping column names
    column_mapping = {
        "hhs_confirmed_admissions_influenza_1d": "flu_hospitalizations",
        "doctor-visits_smoothed_adj_cli": "doctor_visits",
        "hhs_confirmed_admissions_covid_1d": "covid_hospitalizations"
    }

    #Find Columns that are not None in alldata
    matching_columns = [f"{data['data_source'].unique()[0]}_{data['signal'].unique()[0]}"
                        for data in alldata if data is not None and
                        f"{data['data_source'].unique()[0]}_{data['signal'].unique()[0]}" in column_mapping]

    columndb = {f"{col}_{index}_value": column_mapping[col] for index, col in enumerate(matching_columns)}
    missing_columns = pd.DataFrame(columns=[col for col in column_mapping.values() if col not in columndb.values()])
    alldata = [data for data in alldata if data is not None]

    data = covidcast.aggregate_signals(alldata)
    data = pd.concat([data, missing_columns], axis=1)
    data.rename(columns=columndb, inplace=True)

    data = data.rename(columns={"geo_value": "region"})
    data = data.rename(columns={'time_value': 'date'})
    data.reset_index(drop=True, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data['region'] = data['region'].str.upper()
    
    # client = Socrata("healthdata.gov", None)
    # results = client.get("g62h-syeh", limit=100000)
    # df = pd.DataFrame.from_records(results)
    # df['date'] = df['date'].apply(convert_date_format, api=True)
    # df = df[(df['state'] != 'AS') & (df['state'] != 'VI')]

    # # Rename to create consistency
    # df.rename(columns={'state': 'region'}, inplace=True)
    # df = df.rename(columns={"previous_day_admission_influenza_confirmed": "flu_hospitalizations"})
    # df = df.rename(columns={"previous_day_admission_adult_covid_confirmed": "adult_covid"})
    # df = df.rename(columns={"previous_day_admission_pediatric_covid_confirmed": "baby_covid"})
    # df = df.rename(columns={"previous_day_admission_influenza_confirmed": "flu_hospitalizations"})

    # # Aggregate two columns
    # df['flu_hospitalizations'] = pd.to_numeric(df['flu_hospitalizations'], errors='coerce', downcast='integer')
    # df['baby_covid'] = pd.to_numeric(df['baby_covid'], errors='coerce', downcast='integer')
    # df['adult_covid'] = pd.to_numeric(df['adult_covid'], errors='coerce', downcast='integer')
    # df['covid_hospitalizations'] = df['baby_covid'] + df['adult_covid']
    # df = df[['date', 'region', 'flu_hospitalizations', 'covid_hospitalizations']]
    # df['date'] = pd.to_datetime(df['date'])
    
    # columns_to_shift = ['covid_hospitalizations', 'flu_hospitalizations']
    # df[columns_to_shift] = df.groupby('region')[columns_to_shift].shift(periods=-1)
    
    # data = data.merge(df, on=['region', 'date'], how='left')
    # data['flu_hospitalizations'] = data['flu_hospitalizations'].astype('Int64')
    # data['covid_hospitalizations'] = data['covid_hospitalizations'].astype('Int64')
    
    #Fill in Dates missing at the end
    filler_dfs = []

    cumulative_data_state_date = data.groupby(['date']).agg({
        'covid_hospitalizations': 'sum',
        'flu_hospitalizations': 'sum',
    }).reset_index()


    #Create us data from aggregating all the regions
    us_dates = pd.DataFrame({'date': pd.date_range(start_date, end_date), 'region': 'US'})

    #Initalize doctor visits for merge
    us_dates = us_dates.merge(cumulative_data_state_date, on='date', how='left')
    us_dates['doctor_visits'] = np.nan
    data = pd.concat([data, us_dates], ignore_index=True)
    
    #Shift one day forward -> Check on this since Delphi API is shifting one day back.
    columns_to_shift = ['covid_hospitalizations', 'flu_hospitalizations']
    data[columns_to_shift] = data.groupby('region')[columns_to_shift].shift(periods=1)
    
    sub_regions = data['region'].unique()
        
    #Fill in Remaining Days If applicable
    for state in sub_regions:
        state_data = data[data['region'] == state]
        max_date_state = state_data['date'].max() + relativedelta(days=1)
        date_range = pd.date_range(max_date_state, end_date)
        state_dates = pd.DataFrame({'date': date_range, 'region': state, 'covid_hospitalizations': None, 'flu_hospitalizations': None})
        state_dates = state_dates.reindex(columns=data.columns)
        filler_dfs.append(state_dates)
    
    data = pd.concat([data] + filler_dfs, ignore_index=True)
    
    #Fill in first Month
    filler_dfs = []
    for state in sub_regions:
        date_range = pd.date_range(start_date, '2020-02-01')
        state_dates = pd.DataFrame({'date': date_range, 'region': state, 'covid_hospitalizations': None, 'flu_hospitalizations': None})
        state_dates['epiweek'] = state_dates['date'].apply(lambda x: int(Week.fromdate(x, system="cdc").cdcformat()))
        state_dates = state_dates.reindex(columns=data.columns)
        filler_dfs.append(state_dates)
        
    data = pd.concat([data] + filler_dfs, ignore_index=True)

    #Remove Duplicates
    data = data.drop_duplicates(subset=['date', 'region'])
    
    #ADD Fips to data
    data['fips'] = data['region'].map(state_fips) 

    # changes US to 0
    data['fips'] = data['fips'].replace('US', '0')

    #Add Epiweek to data
    data['epiweek'] = data['date'].apply(lambda x: Week.fromdate(x, system="cdc").cdcformat())
    data['epiweek'] = data['epiweek'].astype('int')

    #Keep Relevant Columns
    data = data[[
        "epiweek", "date", "region", "fips", "covid_hospitalizations", "flu_hospitalizations", "doctor_visits"
    ]]

    #Sort by fips and date
    data.sort_values(by=['fips', 'date'], inplace=True)
    data['fips'] = data['fips'].replace('0', 'US')
    
    """
    Step 3: Save daily data
    """
    
    dailysymptom, weeklysymptom = querydata()
    data = data.merge(dailysymptom, on=['date', 'region'], how='left')
    data.drop(['epiweek_y', 'country_region'], axis=1, inplace=True)
    data.rename(columns={'epiweek_x': 'epiweek'}, inplace=True)
    folder_path = "./data/daily/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = f"{folder_path}{epiweek}.csv"
    data.to_csv(path, index=False)
    
    return data, dailysymptom, weeklysymptom, sub_regions
