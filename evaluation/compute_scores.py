import pandas as pd
import os
from sklearn.metrics import mean_absolute_percentage_error
from terminaltables import AsciiTable
from datetime import datetime, timedelta
from sys import *

target = 'SARI'

dataset = pd.read_csv('../../data_for_SFS/SARI/sari.csv', parse_dates=['ds'])
years = sorted(list(set(dataset['year'])))

dataset.dropna(inplace=True)
real_map = dict(zip(dataset['ds'], dataset['avg'+target]))
real = {'ds': list(dataset['ds'])}
real['REAL'] = list(dataset['avg'+target])
real = pd.DataFrame(real)

AGE = ['AGE_jt1', 'AGE_1to4', 'AGE_5to14', 'AGE_15to64', 'AGE_65plus']
LC  = ['FLU', 'RSV', 'RV', 'ENTV', 'ADV', 'HMPV', 'PIV1', 'PIV2', 'PIV3']

covar_combis = [ (target, [target]) ]
covar_combis.append( (target+'+LC', [target] + LC) )
covar_combis.append( (target+'+AGE', [target] + AGE) )
covar_combis.append( (target+'+AGE+LC', [target] + AGE + LC) )

algos = ['TFT','DeepAR','Chronos','AutoML']
if '-test' in argv:
    algos = ['TFT']

pair_of_CIs = []
for ll in [0.05, 0.10, 0.15]:
    uu = 1.0 - ll
    prob = round(uu - ll, 2)
    pair_of_CIs.append( (str(ll), str(uu), prob) )

# 2012 and 2013 were only used within the training data, not for evaluation
years = [2014,2015,2016,2017,2018,2019]
print(years)

# prepare the dates used as earliest days in the evaluation
# this ensures that at least 2 weeks of data are available for a given year
# that are used by the model for the prediction before forecasts are evaluated
min_start_dates = {}
for year in years:
    tempydata = dataset.loc[dataset['year'] == year]
    min_ds = tempydata['ds'].min()
    min_plus_2weeks = min_ds + timedelta(days=14)
    min_start_dates[year] = min_plus_2weeks

plotd = {'Year':[], 'MAPE':[], 'algo':[], 'cov':[], 'algocov':[], 'day':[], 'resolution':[]}

within_CI = {'Year':[], 'lower_CI':[], 'upper_CI':[], 'prob_CI':[],
        'algo':[], 'cov':[], 'day':[], 'within': [], 'resolution':[]}

# prepare also the true (real) data for the line plots shown in the manuscript
real_with_year = real.copy()
real_with_year['year'] = [ ds.year for ds in real['ds'] ]
for year in years:
    avgreal_to_save = {'ds':[], 'REAL':[]}
    yreal = real_with_year.loc[real_with_year['year'] == year].copy()
    for ds in yreal['ds']:
        fromds = ds - timedelta(days=3); tods = ds + timedelta(days=3);
        subset = yreal.loc[(yreal['ds'] >= fromds) & (yreal['ds'] <= tods)].copy()
        avgreal_to_save['ds'].append(ds)
        avgreal_to_save['REAL'].append(subset['REAL'].mean())
    avgreal_to_save = pd.DataFrame(avgreal_to_save)

    # the true data cannot be shared
    #avgreal_to_save.to_csv('./data/avgREAL/y%d.csv'%(year), index=False)

#horizon_days = [1,7,14,21]
horizon_days = list(range(1,22))
trendy_days = [4,7,11,14,18]

for day in horizon_days:
    print('%d-day model:'%(day))
    table = [[''] + [cov_name for cov_name, covars in covar_combis ]]
    for year in years:
        min_start_ds = min_start_dates[year]
        for algo in algos:
            for cov_name, covars in covar_combis:
                algocov = '%s_%s'%(algo, cov_name)

                if algo in ['Chronos','AutoML'] and cov_name != 'SARI':
                    continue
                fp = './concatenated/%s_%s.csv'%(algo, cov_name)
                combined = pd.read_csv(fp, parse_dates=['ds'])
                combined['year'] = [ds.year for ds in combined['ds']]
                ydata = combined.loc[combined['year'] == year].copy()
                ddata = ydata.loc[ydata['day'] == day]
                ddata = ddata.sort_values(by='ds')
                ddata = ddata.merge(real, on='ds')

                # add the data on which the model was trained
                # (based on the simulation)
                ddata['model_ds'] = [ds-timedelta(days=day) for ds in ddata['ds']]
                # remove the predictions for which the model had not enough data
                # from the "current year in the simulation"
                print(ddata['model_ds'])
                print(min_start_ds)
                ddata = ddata.loc[ddata['model_ds'] > min_start_ds]
                print(algo, cov_name, year, day, ddata.shape)

                if algo == 'TFT' and cov_name == 'SARI':
                    # check if real value lies within the CI
                    for ll, uu, prob in pair_of_CIs:
                        within = [ row[ll] <= row['REAL'] and row['REAL'] <= row[uu] for ii, row in ddata.iterrows() ]
                        within_rate = sum(within)/ len(within)
                        print('\t', ll, uu, '%.3f'%(within_rate), prob, sep='\t')

                        {'Year':[], 'lower_CI':[], 'upper_CI':[], 'prob_CI':[],
                                'algo':[], 'cov':[], 'day':[], 'within': []}

                        within_CI['Year'].append(year); within_CI['lower_CI'].append(ll);
                        within_CI['upper_CI'].append(uu); within_CI['prob_CI'].append(prob);
                        within_CI['algo'].append(algo); within_CI['cov'].append(cov_name);
                        within_CI['day'].append(day); within_CI['within'].append(within_rate);
                        within_CI['resolution'].append('D')
                    # use the avg around the days used for the weekly trend predictions
                    if day in trendy_days:
                        fp = './concatenated/AVG_%s_%s.csv'%(algo, cov_name)
                        avg_preds = pd.read_csv(fp, parse_dates=['ds'])
                        dd_avgp = avg_preds.loc[avg_preds['day'] == day].copy()
                        dd_avgp['year'] = [ds.year for ds in dd_avgp['ds']]
                        dd_avgp = dd_avgp.loc[dd_avgp['year'] == year]

                        dd_avgp['model_ds'] = [ds-timedelta(days=day) for ds in dd_avgp['ds']]
                        dd_avgp = dd_avgp.loc[dd_avgp['model_ds'] > min_start_ds]
                        dd_avgp.sort_values(by='ds', inplace=True)
                        temp_map = dict(zip(dd_avgp['ds'], dd_avgp['mean']))

                        avg_pred, avg_real = [], []
                        real_to_merge = {'ds':[], 'REAL':[]}
                        for ds in dd_avgp['ds']:
                            fromds = ds - timedelta(days=3); tods = ds + timedelta(days=3);
                            subset = real.loc[(real['ds'] >= fromds) & (real['ds'] <= tods)].copy()
                            if subset.shape[0] == 7 and ds in temp_map:
                                avg_pred.append( temp_map[ds] )
                                avg_real.append( subset['REAL'].mean() )
                                real_to_merge['ds'].append(ds)
                                real_to_merge['REAL'].append(subset['REAL'].mean())

                        avg_score = mean_absolute_percentage_error(avg_real, avg_pred)
                        real_to_merge = pd.DataFrame(real_to_merge)

                        fp = './data/avgPRED/%s_%s_%d_day%d.csv'%(algo,cov_name,year,day)
                        dd_avgp.to_csv(fp, index=False)

                        dd_avgp = dd_avgp.merge(real_to_merge, on='ds')
                        avg_score = mean_absolute_percentage_error(dd_avgp['REAL'], dd_avgp['mean'])

                        plotd['Year'].append(year); plotd['MAPE'].append(avg_score);
                        plotd['cov'].append(cov_name); plotd['algo'].append(algo);
                        plotd['day'].append(day); plotd['algocov'].append(algocov);
                        plotd['resolution'].append('W')

                        for ll, uu, prob in pair_of_CIs:
                            within = [ row[ll] <= row['REAL'] and row['REAL'] <= row[uu] for ii, row in dd_avgp.iterrows() ]
                            within_rate = sum(within)/ len(within)
                            within_CI['Year'].append(year); within_CI['lower_CI'].append(ll);
                            within_CI['upper_CI'].append(uu); within_CI['prob_CI'].append(prob);
                            within_CI['algo'].append(algo); within_CI['cov'].append(cov_name);
                            within_CI['day'].append(day); within_CI['within'].append(within_rate);
                            within_CI['resolution'].append('W')

                score = mean_absolute_percentage_error(ddata['REAL'], ddata['mean'])
                fp = './data/predictions/%s_%s_%d_day%d.csv'%(algo,cov_name,year,day)
                ddata.to_csv(fp, index=False)

                plotd['Year'].append(year); plotd['MAPE'].append(score);
                plotd['cov'].append(cov_name); plotd['algo'].append(algo);
                plotd['day'].append(day); plotd['algocov'].append(algocov);
                plotd['resolution'].append('D')

plotd = pd.DataFrame(plotd)
print(plotd)
plotd.to_csv('./data/scores_plotd.csv', index=False)

within_CI = pd.DataFrame(within_CI)
print(within_CI)
within_CI.to_csv('./data/within_CI_rates.csv', index=False)



