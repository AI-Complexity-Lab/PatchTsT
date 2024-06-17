from datetime import datetime
import pandas as pd
import os
from sodapy import Socrata
from epiweeks import Week
from datetime import date


def convert_date_format(date_str, api):
    if api:
        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
    return date_obj.strftime('%Y-%m-%d')

client = Socrata("healthdata.gov", None)
results = client.get("g62h-syeh", limit=100000)
df = pd.DataFrame.from_records(results)
df['date'] = df['date'].apply(convert_date_format, api=True)
df = df[(df['state'] != 'AS') & (df['state'] != 'VI')]

# Rename to create consistency
df.rename(columns={'state': 'region'}, inplace=True)
df = df.rename(columns={"previous_day_admission_influenza_confirmed": "flu_hospitalizations"})
df = df.rename(columns={"previous_day_admission_adult_covid_confirmed": "adult_covid"})
df = df.rename(columns={"previous_day_admission_pediatric_covid_confirmed": "baby_covid"})
df = df.rename(columns={"previous_day_admission_influenza_confirmed": "flu_hospitalizations"})

# Aggregate two columns
df['flu_hospitalizations'] = pd.to_numeric(df['flu_hospitalizations'], errors='coerce', downcast='integer')
df['baby_covid'] = pd.to_numeric(df['baby_covid'], errors='coerce', downcast='integer')
df['adult_covid'] = pd.to_numeric(df['adult_covid'], errors='coerce', downcast='integer')
df['covid_hospitalizations'] = df['baby_covid'] + df['adult_covid']
df = df[['date', 'region', 'flu_hospitalizations', 'covid_hospitalizations']]
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['region', 'date'], inplace=True)

filler_dfs = []
sub_regions = df['region'].unique()
start_date = date(2020, 1, 1)
for state in sub_regions:
    date_range = pd.date_range(start_date, '2020-02-01')
    state_dates = pd.DataFrame({'date': date_range, 'region': state, 'covid_hospitalizations': None, 'flu_hospitalizations': None})
    state_dates['epiweek'] = state_dates['date'].apply(lambda x: int(Week.fromdate(x, system="cdc").cdcformat()))
    state_dates = state_dates.reindex(columns=df.columns)
    filler_dfs.append(state_dates)
    
data = pd.concat([df] + filler_dfs, ignore_index=True)

#Remove Duplicates
data = data.drop_duplicates(subset=['date', 'region'])
data.to_csv('test.csv', index=False)