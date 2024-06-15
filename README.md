# SFS
### Code and experiemental results for the SARI Forecasting System

The script `model_training_forecasting.py` is applied for a combination of each algorithm, split date, and set of covariates. The split date defines the date on which the time series is split into training and testing data. The following example trains a TFT model on all data available until July 17, 2019 (inclusive), using the laboratory component (LC) covariates.  
```
python model_training_forecasting.py 2019-07-17 LC TFT
```
Two example outputs for such an execution are provided in the folder `out_TFT`. 

The script `create_walk-forward_executions_NeSI.py` has been implemented to create all combinations and creates lists of command line calls that are used to submit job arrays on the NeSI high-performance computing cluster. See the files in `NeSI_execs` for all the executions.

The forecasts resulting from the one-day walk-forward validation are first concatenated by the script `evaluation/collect_and_concatenate.py`. During the concatenation, the weekly trend forecast is created by averaging the daily forecasts. Concatenated files are written into the folder `concatenated`, providing all forecasts summarized by algorithm and covariate set.

The script `compute_scores.py` is then used to compute the evaluation metric (MAPE) for the machine learning benchmark. 

The folder `pearson_correlation_analysis` provides the Python script needed to run the Pearson correlation analysis implemented for the laboratory component. This script also creates the heatmap shown in the manuscript. 



