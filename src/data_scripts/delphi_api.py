import covidcast
from datetime import date
from epiweeks import Week, Year
import pandas as pd
# import pdb

## BELOW IS THE EXAMPLE CODE - update your own api key
#################################################################
#################################################################

states = pd.read_csv('./data/states.csv', header=0).iloc[:, 1].astype(str)
# convert states to lowercase - required to input states to covidcast
# see https://cmu-delphi.github.io/delphi-epidata/api/covidcast_geography.html
states = states.str.lower().to_list()

# query covidcast API for data
# Note:
#   smoothed_adj_cli is https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/doctor-visits.html
#   wcli (since April 15) from https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html
#   and many more if we use since 2020-09-08
covidcast.use_api_key("163cdbefcb55b")
start_date, end_date = date(2020, 1, 1), date(2022, 1, 1)

doctor = covidcast.signal(
    "doctor-visits",
    "smoothed_adj_cli",  # smoothed_cli
    start_date,
    end_date,
    "state",
    geo_values=states)
fb_wili = covidcast.signal("fb-survey",
                           "smoothed_wili",
                           start_date,
                           end_date,
                           "state",
                           geo_values=states)
fb_wcli = covidcast.signal("fb-survey",
                           "smoothed_wcli",
                           start_date,
                           end_date,
                           "state",
                           geo_values=states)
deaths = covidcast.signal("indicator-combination",
                          "deaths_incidence_num",
                          start_date,
                          end_date,
                          "state",
                          geo_values=states)
cases = covidcast.signal("indicator-combination",
                         "confirmed_incidence_num",
                         start_date,
                         end_date,
                         "state",
                         geo_values=states)

data = covidcast.aggregate_signals(
    [cases, deaths, doctor, fb_wili, fb_wcli, doctor])
data.columns

data = covidcast.aggregate_signals([cases, deaths, doctor, fb_wili, fb_wcli])
data = data.rename(
    columns={
        "indicator-combination_confirmed_incidence_num_0_value": "cases",
        "indicator-combination_deaths_incidence_num_1_value": "deaths",
        "doctor-visits_smoothed_adj_cli_2_value": "doctor_visits",
        "fb-survey_smoothed_wili_3_value": "wili",
        "fb-survey_smoothed_wcli_4_value": "wcli"
    })

print(data[[
    "time_value", "geo_value", "cases", "deaths", "doctor_visits", "wili",
    "wcli"
]].head(20))
# data.head()