import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('../../data_for_SFS/SARI/sari.csv', parse_dates=['ds'])

# all in mm:
mm_per_inch = 25.4
bmc_width_1 = 85
bmc_width_2 = 170
max_height = 225
max_height_inch = max_height / mm_per_inch
width_inch = bmc_width_2 / mm_per_inch

days_back = [0, 3, 7, 10, 14, 18, 21]

fs_ticks, fs_label, fs_title = 12, 12, 14
plt.rcParams['figure.figsize'] = [5, 5]
lw = 3.0
cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
cmap = 'BrBG'
cmap = sns.diverging_palette(230, 20, as_cmap=True)

viruses =  ['flu', 'rsv', 'rv', 'adv', 'entv', 'hmpv']
viruses += ['piv1', 'piv2', 'piv3']
upper_viruses = [ v.upper() for v in viruses ]
cois = [ 'avg'+v.upper() for v in viruses ]

for col in cois + ['avgSARI']:
    dataset.dropna(subset=[col], axis=0, inplace=True)

years = [2012,2013,2014,2015,2016,2017,2018,2019]
day_labels = {0:'no shift'}

xlab_order = []
pivots = {}

for vname, col in zip(upper_viruses, cois):
    plotd = {'year':[], 'days':[], 'PCC':[]}
    # iterate over years
    for year in years:
        # for each year, create a subset of the data as basis for the correlation analysis
        y_data = dataset.loc[dataset['year'] == year].copy()
        y_data.sort_values(by='ds', inplace=True)
        sari = list(y_data['avgSARI']); virus = list(y_data[col]);

        # apply a time shift to the virus incidence and compute the correlation coefficient
        for db in days_back:
            temp = {'sari': sari+db*[np.nan], 'virus': db*[np.nan]+virus}
            temp = pd.DataFrame(temp)
            temp.dropna(subset=['sari','virus'], axis=0, inplace=True)
            pcc = pearsonr(temp['sari'], temp['virus'])[0]
            print('\t', '%.4f'%pcc)
            xlab = day_labels.get(db, '-%d days'%(db))

            # collect the results in a dictionary
            plotd['year'].append(year)
            plotd['days'].append( xlab )
            plotd['PCC'].append(pcc)
            if not xlab in xlab_order:
                xlab_order.append(xlab)

    # convert the dictionary into the right data format for the heatmap
    plotd = pd.DataFrame(plotd)
    noNA_plotd = plotd.dropna()
    mini, maxi = noNA_plotd['PCC'].min(), noNA_plotd['PCC'].max()
    abs_max = max([abs(mini), abs(maxi)])

    plotd = plotd.pivot("year", "days", "PCC")
    plotd = plotd.reindex(xlab_order, axis=1)
    print(plotd)

    pivots[vname] = (plotd, abs_max)

    plt.rcParams['figure.figsize'] = [5, 5]
    ax = sns.heatmap(plotd, annot=True, vmin=-abs_max, vmax=abs_max,
        cmap=cmap)
    plt.title(col[3:], fontsize=fs_title, fontweight='bold')
    plt.xticks(fontsize=fs_ticks); plt.yticks(fontsize=fs_ticks);
    plt.yticks(rotation=0)
    ax.collections[0].colorbar.set_label('PCC with SARI (no shift)',
        fontsize=fs_label)
    ax.set_xlabel('Days back in time',fontsize=fs_label)
    ax.set_ylabel('', fontsize=0.0001)
    plt.tight_layout()
    plt.savefig('./heatmaps/%s.png'%(vname))
    plt.clf()

# from here: create a multi-panel plot with 3x3 heatmaps
height_inch = width_inch *0.95
plt.rcParams['figure.figsize'] = [width_inch, height_inch]
fig = plt.figure()

fs_ticks, fs_label, fs_title = 6, 7, 8

for pi, vname in enumerate(upper_viruses):
    plotd, abs_max = pivots[vname]
    abs_max = 0.82
    show_cbar = (pi+1) % 3 == 0
    #show_cbar = False

    plt.subplot(3, 3, pi+1)
    ax = sns.heatmap(plotd, annot=True, vmin=-abs_max, vmax=abs_max,
        cmap=cmap, annot_kws={"size": fs_ticks-2}, cbar=show_cbar)
    plt.title(vname, fontsize=fs_title, fontweight='bold')
    plt.xticks(fontsize=fs_ticks+1, rotation = 30)
    plt.yticks(fontsize=fs_ticks+1, rotation=0)

    if (pi+1) % 3 == 0 and show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.set_label('PCC with SARI (no shift)', fontsize=fs_label)
        cbar.ax.tick_params(labelsize=fs_ticks)
    if (pi+1) % 3 != 1:
        plt.yticks([i+0.5 for i in range(8)], 8*[''], fontsize=0.00001)
    if vname in ['PIV1', 'PIV2', 'PIV3']:
        ax.set_xlabel('Days back in time',fontsize=fs_label+2)
    else:
        ax.set_xlabel('',fontsize=0.0001)
        plt.xticks([i+0.5 for i in range(7)], 7*[''], fontsize=0.00001)
    ax.set_ylabel('', fontsize=0.0001)
plt.tight_layout()
plt.savefig('./multipanel_heatmaps.png')
plt.savefig('./multipanel_heatmaps.svg')
plt.clf()















