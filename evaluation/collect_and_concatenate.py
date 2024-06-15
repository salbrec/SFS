import pandas as pd
import os
from datetime import datetime, timedelta
from sys import *

target = 'SARI'

dataset = pd.read_csv('../../data_for_SFS/SARI/sari.csv', parse_dates=['ds'])
years = [2014,2015,2016,2017,2018,2019]

algos = ['AutoML', 'TFT','DeepAR','Chronos']
if '-test' in argv:
    algos = ['Chronos']
covars = ['NONE', 'AGE', 'LC', 'AGE+LC']
better_names = ['SARI', 'SARI+AGE', 'SARI+LC', 'SARI+AGE+LC']

# create day range required to derive the weekly trend forcast
day_ranges = {7:set([4,5,6,7,8,9,10]), 14:set([11,12,13,14,15,16,17])}
day_ranges.update( {4:set([1,2,3,4,5,6,7])} )
day_ranges.update( {11:set([8,9,10,11,12,13,14])} )
day_ranges.update( {18:set([15,16,17,18,19,20,21])} )
trendy_days = [4,7,11,14,18]

# iterate over algorithms and covariate settings to collect and concatenate
# all predictions from the one-day walk-forward validation
for algo in algos:
    for cov, cov_name in zip(covars, better_names):
        fp_out = './concatenated/%s_%s.csv'%(algo, cov_name)
        if os.path.exists(fp_out):
            continue
        if algo in ['Chronos', 'AutoML'] and cov != 'NONE':
            continue
        avg_forecast, aggr_cols = {}, []
        covar_preds = '../out_%s/MSE/SARI/SARI/%s/'%(algo, cov)
        print(covar_preds)
        concat = []
        for subdir, dirs, files in os.walk(covar_preds):
            fp = subdir + '/predictions.csv'
            if os.path.exists(fp):
                temp = pd.read_csv(fp, parse_dates=['ds'])
                temp['day'] = [ ii+1 for ii in range(temp.shape[0]) ]
                concat.append(temp)

                # compute and add averaged predictions
                if len(avg_forecast) == 0:
                    for col in temp.columns:
                        avg_forecast[col] = []
                        if not col in ['ds', 'item', 'day']:
                            aggr_cols.append(col)
                day_ds = dict(zip(temp['day'], temp['ds']))
                for day in trendy_days:
                    selection = [ dd in day_ranges[day] for dd in temp['day'] ]
                    around_day = temp.loc[selection].copy()
                    avg_forecast['ds'].append(day_ds[day])
                    avg_forecast['item'].append('SARI')
                    for col in aggr_cols:
                        avg_forecast[col].append( around_day[col].mean() )
                    avg_forecast['day'].append(day)

        combined = pd.concat(concat)
        print(combined)
        combined.to_csv(fp_out,index=False)

        avg_forecast = pd.DataFrame(avg_forecast)
        fp_out = './concatenated/AVG_%s_%s.csv'%(algo, cov_name)
        avg_forecast.to_csv(fp_out,index=False)
        print('Done with', algo, cov_name)
    print()









