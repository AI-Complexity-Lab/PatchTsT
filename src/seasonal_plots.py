import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from epiweeks import Week
import argparse

#Smooth Yearly Data For A Region
def smooth_data(data, column, window=3):
    smoothed_data = data.copy()
    smoothed_data['smoothed_' + column] = data[column].rolling(window=window, min_periods=1).mean()
    return smoothed_data

#Create Plot For Region
def create_region_plot(region_data, common_ymax, region, output_directory, columndate, seasons):
    fig, axs = plt.subplots(figsize=(10, 5), nrows=3, sharex=False)
    fig.suptitle(f'Flu Hospitalizations in {region}')
    
    for i, year in enumerate(seasons):
        start_date = pd.to_datetime(f'{year}-10-01')
        end_date = pd.to_datetime(f'{year + 1}-09-30')
        year_data = region_data[(region_data[columndate] >= start_date) & (region_data[columndate] <= end_date)]
        targetcolumns = "flu_hospitalizations"
        if columndate == "date":
            year_data = smooth_data(year_data, 'flu_hospitalizations', window=7)
            targetcolumns = "smoothed_flu_hospitalizations"
        ax = axs[i]
        ax.plot(year_data[columndate], year_data[targetcolumns])
        ax.set_ylabel('Flu Hospitalizations')
        ax.set_title(f'Year {year}-{year + 1}')
        ax.grid(True)
        
        ax.set_xlim(start_date, end_date)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_ylim(0, common_ymax)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(output_directory, f'{region}_plots.png'))
    plt.close()


#Plot Geographical Regions:
def plotgeoregions(seasonaldata, output_directory, graphtype, columndate, seasons):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    state_to_region = {
        'AL': 'South', 'AK': 'West', 'AZ': 'West', 'AR': 'South','CA': 'West',
        'CO': 'West', 'CT': 'NorthEast', 'DE': 'NorthEast', 'FL': 'South',
        'GA': 'South', 'HI': 'West', 'ID': 'West', 'IL': 'Midwest', 'IN': 'Midwest',
        'IA': 'Midwest', 'KS': 'Midwest', 'KY': 'South', 'LA': 'South', 'ME': 'NorthEast',
        'MD': 'South', 'MA': 'NorthEast', 'MI': 'Midwest', 'MN': 'Midwest', 'MS': 'South',
        'MO': 'Midwest', 'MT': 'West', 'NE': 'Midwest', 'NV': 'West', 'NH': 'NorthEast',
        'NJ': 'NorthEast', 'NM': 'West', 'NY': 'NorthEast', 'NC': 'South', 'ND': 'Midwest',
        'OH': 'Midwest', 'OK': 'South', 'OR': 'West', 'PA': 'NorthEast', 'RI': 'NorthEast',
        'SC': 'South', 'SD': 'Midwest', 'TN': 'South', 'TX': 'South', 'UT': 'West', 'VT': 'NorthEast',
        'VA': 'South', 'WA': 'West', 'WV': 'South', 'WI': 'Midwest', 'WY': 'West', 'PR': 'South'
    }
    seasonaldata['georegion'] = seasonaldata['region'].map(state_to_region)
    seasonaldata = seasonaldata.groupby('georegion')

    for region, region_data in seasonaldata:
        if graphtype == 0:
            region_data = region_data.groupby(columndate).sum(numeric_only=True).reset_index()
        else:
            region_data = region_data[[columndate, 'flu_hospitalizations']]
        max_values = []
        for year in seasons:
            start_date = pd.to_datetime(f'{year}-10-01')
            end_date = pd.to_datetime(f'{year + 1}-09-30')
            season_data = region_data[(region_data[columndate] >= start_date) & (region_data[columndate] <= end_date)]
            max_values.append(season_data['flu_hospitalizations'].max())
        common_ymax = max(max_values)
        create_region_plot(region_data, common_ymax, region, output_directory, columndate, seasons)


#Plot States:
def plotstates(seasonaldata, output_directory, columndate, seasons):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    sub_regions = seasonaldata['region'].unique()
    for region in sub_regions:
        region_data = seasonaldata[seasonaldata['region'] == region][[columndate, 'flu_hospitalizations']]
        max_values = []
        for year in seasons:
            start_date = pd.to_datetime(f'{year}-10-01')
            end_date = pd.to_datetime(f'{year + 1}-09-30')
            season_data = region_data[(region_data[columndate] >= start_date) & (region_data[columndate] <= end_date)]
            max_values.append(season_data['flu_hospitalizations'].max())
        common_ymax = max(max_values)
        create_region_plot(region_data, common_ymax, region, output_directory, columndate, seasons)

if __name__ == '__main__':
    # Generate State-wise or Regional Plots from Daily or Weekly Data
    # Regional Plots can be aggregated or be consisted of individual state-wise plots within region
    # Use from Command Like As Shown Below:
    # --plot states --type [daily/weekly]
    # --plot geo --geo [0/1] --type [daily/weekly]
    
    #Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', choices=['state', 'geo'], help='Plot states or geographical regions')
    parser.add_argument('--type', choices=['daily', 'weekly'], help='Plot daily or weekly data')
    parser.add_argument('--geo', nargs='?', type=int, help='Plot Combined (0) or Individual (1)')
    args = parser.parse_args()
    
    #Find Last File
    epiweek = Week.thisweek(system="CDC")-1
    directory_path = f"./data/{args.type}"
    columndate = "date"
    if args.type == "weekly":
        directory_path += "/weeklydata"
        columndate = "end_date"
    files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    files.sort()
    lastfile = files[-1]

    #Read Data and Create Folder
    seasonaldata = pd.read_csv(f"{directory_path}/{lastfile}")
    seasonaldata[columndate] = pd.to_datetime(seasonaldata[columndate])
    
    #seasons to plot:
    seasons = [2021,2022,2023]
    #Call Appropriate Function
    if args.plot == 'state':
        output_directory = f"plotstates/{epiweek}/{args.type}"
        plotstates(seasonaldata, output_directory, columndate, seasons)
    elif args.plot == 'geo':
        if args.geo == 0:
            output_directory = f"plotregions/{epiweek}/{args.type}/combined"
        else:
            output_directory = f"plotregions/{epiweek}/{args.type}/individual"
        plotgeoregions(seasonaldata, output_directory, args.geo, columndate, seasons)
