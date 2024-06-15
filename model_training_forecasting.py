import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import subprocess
from sys import *
from datetime import datetime, timedelta
from copy import deepcopy

from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.evaluation.metrics import mse, mase
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

dataset = pd.read_csv('../data_for_SFS/SARI/sari.csv', parse_dates=['ds'])
print(dataset)

# create the values for 19 quantile forecasts to be explored potentially
lquant, uquant = [], []
for qq in range(5,50,5):
    lquant.append(qq/100); uquant.append(1.0 - qq/100);
quant_levels = lquant + [0.5] + sorted(uquant)
print('Create these quantile levels:', quant_levels)

# these are the settings used as command line arguments on NeSI
# example call: "python model_training_forecasting.py 17-07-2019 NONE Chronos"
split_date = argv[1]    # defines the date used to split into training and testing set
covars = argv[2]        # covariates used:  NONE, AGE, LC, AGE+LC
algo = argv[3]          # algorithm used: TFT, DeepAR, AutoML, Chronos
sdds = datetime.strptime(split_date, '%Y-%m-%d')

# some default settings
eval_metric = 'MSE'
data_set = 'SARI'
target = 'SARI'

preset_used = None
time_limit = 150000

# prepare the list of time series variables that need to be integrated in the dataset
all_vars = [target]
if covars != 'NONE':
    all_vars += covars.split('+')
if 'AGE' in covars:
    all_vars.remove('AGE')
    all_vars += ['AGE_jt1', 'AGE_1to4', 'AGE_5to14', 'AGE_15to64', 'AGE_65plus']
if 'LC' in covars:
    all_vars.remove('LC')
    all_vars += ['FLU', 'RSV', 'RV', 'ENTV', 'ADV', 'HMPV', 'PIV1', 'PIV2', 'PIV3']

# define a SEED, the forecsasting horizon and the output folder (depending on the algorithm)
SEED = 482020
horizon = 21
base_dir = 'out_' + algo
out_path = os.path.join(base_dir, eval_metric, data_set,
                target, covars, split_date)
print('These are the vars:', all_vars)

# remove rows that have NANs in one of the important columns
# the prefix "avg" specifies columns in which SARI cases were averaged based
# on the 7-day sliding window as described in the manunscript
col_prefix = 'avg'
for col in all_vars:
    dataset.dropna(subset=[col_prefix+col], axis=0, inplace=True)

# create a new DataFrame having averything needed
data = {'start':list(dataset['ds'])}
data['item_id'] = dataset.shape[0] * [target]
for col in all_vars:
    data[col] = list(dataset['avg' + col])
data = pd.DataFrame(data)
temp_train = data.loc[data['start'] <= split_date]

# fill the summer gap with 0s as described in the manuscript
print('Filling with 0s now...')
min_ds, max_ds = temp_train['start'].min(), temp_train['start'].max()
print(min_ds, max_ds)
filled_train = {'start':None, 'item_id':None}
filling_value = 0.00001
for col in temp_train.columns:
    if col in filled_train:
        continue
    value_map = dict(zip(temp_train['start'], temp_train[col]))
    running_ds = deepcopy(min_ds)
    new_ds, temp_col = [], []
    while running_ds <= max_ds:
        temp_col.append( value_map.get(running_ds, filling_value) )
        new_ds.append(running_ds)
        running_ds += timedelta(days=1)
    filled_train[col] = temp_col
    if filled_train['start'] == None:
        filled_train['start'] = new_ds
filled_train['item_id'] = len(filled_train['start']) * [target]
filled_train = pd.DataFrame(filled_train)

# prepare a sequencially arranged dataset for DeepAR
# this is required by DeepAR for the multivariate-to-multivariate forecast
sequential = {'start':[], 'item_id':[], 'TS':[]}
for col in filled_train.columns:
    if col in ['start','item_id']:
        continue
    sequential['start'] += list(filled_train['start'])
    sequential['item_id'] += filled_train.shape[0] * [col]
    sequential['TS'] += list(filled_train[col])
sequential = pd.DataFrame(sequential)

train_data = None
if algo == 'DeepAR':
    train_data = TimeSeriesDataFrame.from_data_frame(sequential,
    id_column="item_id", timestamp_column="start")
else:
    train_data = TimeSeriesDataFrame.from_data_frame(filled_train,
    id_column="item_id", timestamp_column="start")

# convert training data into the format required for autogluon
train_data = train_data.convert_frequency('D')

predictor = None
if algo == 'TFT':
    # define the basic predictor
    predictor = TimeSeriesPredictor(prediction_length=horizon, path=out_path,
                    target=target, eval_metric=eval_metric, quantile_levels=quant_levels)

    agts_algo = 'TemporalFusionTransformerModel'
    # try to set up the transformer
    predictor.fit(train_data, time_limit=time_limit,
        verbosity=4, random_seed=SEED,
        hyperparameters={agts_algo: {}})
if algo == 'DeepAR':
    predictor = TimeSeriesPredictor(prediction_length=horizon, path=out_path,
        target='TS', eval_metric=eval_metric, quantile_levels=quant_levels)

    agts_algo = 'DeepAR'
    predictor.fit(train_data, time_limit=time_limit,
        verbosity=4, random_seed=SEED, hyperparameters={agts_algo: {}})
if algo == 'AutoML':
    predictor = TimeSeriesPredictor(prediction_length=horizon, path=out_path,
        target=target, eval_metric=eval_metric, quantile_levels=quant_levels)
    preset_used = 'best_quality' # it should be "best_quality", use "fast_training" for testing
    predictor.fit(train_data, presets=preset_used, time_limit=time_limit,
        verbosity=4, random_seed=SEED)
if algo == 'Chronos':
    predictor = TimeSeriesPredictor(prediction_length=horizon, path=out_path,
        target=target, eval_metric=eval_metric, quantile_levels=quant_levels)
    preset_used = 'chronos_large' # it should be "chronos_large"
    predictor.fit(train_data, presets=preset_used, time_limit=time_limit,
        verbosity=4, random_seed=SEED)

predictions = predictor.predict(train_data, random_seed=SEED)
print(predictions)

# save agts predictions
pred_fp = out_path + '/agts_predictions.pkl'
pickle.dump(predictions, open(pred_fp, 'wb'))
print('AG-TS pred: ', list(predictions['mean']))

# write out main information
summary = predictor.fit_summary(verbosity=4)

leaderboard_train = summary['leaderboard']
out_fp = out_path + '/train_LB.csv'
leaderboard_train.to_csv(out_fp)

out_fp = out_path + '/train_summary.pkl'
pickle.dump(summary, open(out_fp, 'wb'))

# delete the saved models to save space on the project folder
model_dir = out_path + '/models/'
os.system('rm %s -r'%(model_dir))

# convert predictions to a simpler DataFrame structure
new_dates, items = [], []
for ii in predictions.index:
    items.append(ii[0])
    new_dates.append(ii[1])
preds_df_all = {'ds': new_dates, 'item':items}
for col in predictions.columns:
    preds_df_all[col] = list(predictions[col])
preds_df_all = pd.DataFrame(preds_df_all)
preds_df = preds_df_all.loc[preds_df_all['item'] == target]
pred_fp = out_path + '/predictions.csv'
preds_df.to_csv(pred_fp, index=False)
print(preds_df)












