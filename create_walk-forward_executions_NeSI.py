import pandas as pd
import numpy as np
import pickle
import random
from datetime import timedelta
from os.path import exists


data = pd.read_csv('../data_for_SFS/SARI/sari.csv', parse_dates=['ds'])
years = [2014,2015,2016,2017,2018,2019]

cols = list(filter(lambda x: 'avg' in x, data.columns))
data = data.dropna(subset=cols)

covars = ['NONE','AGE','AGE+LC','LC']

algos = ['TFT', 'DeepAR', 'Chronos', 'AutoML']

# first collect all valid dates from the dataset
# each date will be used once to create a training and testing set
all_split_dates = []
for year in years:
    ydata = data.loc[data['year'] == year].copy()
    min_ds, max_ds = ydata['ds'].min(), ydata['ds'].max()
    split_date = min_ds
    stop_date = max_ds

    while split_date <= stop_date:
        split_date_str = split_date.strftime("%Y-%m-%d")
        all_split_dates.append( split_date_str )
        split_date += timedelta(days=1)

# for each combination of algorithm, split date and covariate setting,
# create a line that executes the model training and forecasting for
# job arrays on NeSI
for algo in algos:
    runs = ''
    for sdate in all_split_dates:
        for cov in covars:
            if algo in ['Chronos', 'AutoML'] and cov != 'NONE':
                continue
            runs += 'srun --unbuffered python model_training_forecasting.py %s %s %s\n'%(sdate, cov, algo)
    open('./NeSI_execs/%s_execs'%(algo), 'w').write(runs)



