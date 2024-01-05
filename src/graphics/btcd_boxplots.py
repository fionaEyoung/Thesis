import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib import colormaps, cm, rcdefaults
import seaborn as sns
from os import path, getlogin, pardir
import os, sys, argparse
import json
from itertools import combinations
import pandas as pd
from math import factorial, ceil
import re
import pickle
from figutils import *


METHODS = ['TF', 'TSD', 'TG']
HEMS = ['l', 'r']

def main():
  current_dir = path.dirname(__file__)
  data_dir = path.join(current_dir, 'raw_results')
  results_dir = path.join(current_dir, pardir, 'figs', 'chapter_5')

  include_metrics = ['binary_dice','density_correlation','ba_schilling_volume', 'ba_schilling_signed_m1']
  metadatfile = path.join(data_dir, 'btcd_cohort.csv')
  meta = pd.read_csv(metadatfile).set_index('participant_id')

  ## Part 1: Box plots (all data)

  filename = 'btcd_box'
  datasets = ['btcd_filtered']

  TRACTS = ['af', 'cst', 'ifo', 'or']
  compare_against = 'TG'
  order = METHODS.copy()
  order.remove(compare_against)

  all_results = pd.DataFrame()
  for dataset in datasets:
    datafile = f'{dataset}.pkl'
    all_results = pd.concat((all_results, pd.read_pickle(path.join(data_dir, datafile))))

  all_results['subject'] = all_results.method1.apply(lambda x: re.findall(r'sub-PAT\d{2}', x['subject'])[0])
  all_results['ipsi'] = ( (meta.loc[all_results['subject'],'hemisphere'].reset_index(drop=True)==all_results['hem'].reset_index(drop=True))
                        | (meta.loc[all_results['subject'],'hemisphere'].reset_index(drop=True)=='c') )
  all_results['def'] = meta.loc[all_results['subject'],'deformation modelling'].reset_index(drop=True)

  n = len(METHODS)
  n_m = len(include_metrics) # nr of metrics
  n_c = n-1 # nr of comparisons
  n_t = len(TRACTS) # nr of tracts

  METRIC_PARAMS['ba_schilling_signed_m1']['lims'] = [-15,15]
  METRIC_PARAMS['ba_schilling_volume']['lims'] = [0,12]

  ## All metrics box plots, grouped by tract
  all_metrics_fig, all_metrics_axs = plt.subplots(nrows=2,
                                                  ncols=ceil(len(include_metrics)/2),
                                                  layout='constrained',
                                                  figsize=set_size(subplots=(2,2)))

  for metric, ax in zip(include_metrics, all_metrics_axs.flat):

      x = np.arange(1, (n_c+1)*n_t, (n_c+1))
      one_of_each = []

      for i, method in enumerate(order):
          mask = ((all_results['methods'] == {compare_against, method}) & (all_results['tract'].isin(TRACTS)))
          # defmask = (mask & all_results['def'] & all_results['ipsi'])
          if not metric == 'ba_schilling_signed_m1':
              box = all_results[mask].boxplot(column=metric, by='tract', ax=ax, grid=False,
                                        return_type='dict', patch_artist=True,
                                        flierprops=flierprops, showcaps=False,
                                        boxprops=boxprops, whiskerprops=boxlineprops, medianprops=boxlineprops,
                                        # boxprops=dict(hatch='////'),
                                        positions=x+i)
              set_box_colours(box[metric], color=METHOD_PROPS[method]['color'])
              # set_box_colours(box[metric], hatch=METHOD_PROPS[method]['hatch'], color='k', facecolor='w')
              one_of_each.append(box[metric]['boxes'][0])
              ## Try and plot the datapoints for deformed subjects only (but I think this also includes postop)
              # for t in TRACTS:
              #   all_results[defmask & (all_results['tract']==t)].plot(kind='scatter', ax=ax, x='tract', y=metric, s=1)

          else:
              # special case for signed
              box = all_results[mask].assign(
                  signed=lambda df: np.where(df.method1.apply(lambda x: x['name']) == method, df.ba_schilling_signed_m1, df.ba_schilling_signed_m2)
              ).boxplot(column='signed', by='tract', ax=ax, grid=False,
                        return_type='dict', patch_artist=True,
                        flierprops=flierprops, showcaps=False,
                        boxprops=boxprops, whiskerprops=boxlineprops, medianprops=boxlineprops,
                        positions=x+i)
              set_box_colours(box['signed'], color=METHOD_PROPS[method]['color'])
              # set_box_colours(box['signed'], hatch=METHOD_PROPS[method]['hatch'], color='k', facecolor='w')
              one_of_each.append(box['signed']['boxes'][0])

      if metric == 'ba_schilling_signed_m1':
          ax.axhline(y=0, c='#8a8a8a', ls='--', linewidth=0.5)

      ax.set_xticks(x+1)
      ax.set_xticklabels(['IFOF' if t.get_text()=='ifo' else t.get_text().upper() for t in ax.get_xticklabels()])
      units = METRIC_PARAMS[metric].get('units')
      ax.set_ylabel(METRIC_PARAMS[metric]['title']+(f' ({units})' if units else ''))
      ax.set_xlabel('')
      ax.tick_params(axis='x', length=0)
      ax.set_ylim(METRIC_PARAMS[metric]['lims'])
      ax.set_title("")

  l = all_metrics_axs[0,0].legend(one_of_each, order, ncol=len(order), loc='lower center')
  l.get_frame().set_linewidth(lw)
  all_metrics_fig.suptitle("BTCD")
  plt.margins(0,0)
  all_metrics_fig.savefig(path.join(results_dir, f'{filename}.pdf'),
              transparent=False, dpi=80)

  ## PART 2: Change plot

  rcdefaults()
  plt.rcParams.update(tex_fonts)
  plt.rcParams.update({"legend.framealpha": 1})

  filename = 'btcd_defchange'
  datafile = 'btcd_def_filtered.pkl'
  all_results = pd.read_pickle(path.join(data_dir, datafile))
  all_results['subject'] = all_results.method1.apply(lambda x: re.findall(r'sub-PAT\d{2}', x['subject'])[0])
  all_results['ipsi'] = ( (meta.loc[all_results['subject'],'hemisphere'].reset_index(drop=True)==all_results['hem'].reset_index(drop=True))
                        | (meta.loc[all_results['subject'],'hemisphere'].reset_index(drop=True)=='c') )
  all_results['def'] = meta.loc[all_results['subject'],'deformation modelling'].reset_index(drop=True)
  all_results['t_side'] = meta.loc[all_results['subject'],'hemisphere'].reset_index(drop=True)

  metric = 'density_correlation'
  change_plot_fig, ax = plt.subplots(layout='constrained',
                                     figsize=set_size(ratio=1.5, fraction=0.5))
  fancy_marker_style = dict(marker='o', linestyle='-', linewidth=2*lw, markersize=5*ms,
                             color='k', markeredgewidth=lw,
                             markerfacecoloralt='white',
                             markeredgecolor='k')
  fill_styles = {'l':'left', 'r':'right', 'c':'full'}
  single_marker_style = dict(c='#C3C3C3', markeredgecolor='#C3C3C3',
                              marker='o', markeredgewidth=lw, markersize=2*ms,
                              linewidth=lw, linestyle='-')
  # for tract in TRACTS:
  for ipsiside in [True, False]:
    # mask = ((#(all_results['methods'] == {compare_against, 'TF'}) | (all_results['methods'] == {compare_against, 'TFD'}) ) &
             # (all_results['tract'] == tract ) &
    mask = ((all_results['ipsi'] == ipsiside ))

    x = np.hstack((#all_results[mask & (all_results['methods']=={'TSD','TG'})][[metric, 'subject']].groupby('subject').mean(),
                   all_results[mask & (all_results['methods']=={'TF','TG'})][[metric]],
                   all_results[mask & (all_results['methods']=={'TFD','TG'})][[metric]]))

    # ax.plot([1,2], x.T, 'X-' if ipsiside else 'o-', c='#AAAAAA',
    #         markersize=2*ms, linewidth=lw, alpha=0.7, markeredgecolor='none')
    ax.plot([1,2], x.T, markerfacecolor = '#C3C3C3' if ipsiside else 'white', **single_marker_style)

  for ipsiside in [True, False]:
    for side in ['r', 'l', 'c']:

      mask = ((all_results['ipsi'] == ipsiside) & (all_results['t_side'] == side))

      x = np.hstack((all_results[mask & (all_results['methods']=={'TF','TG'})][[metric, 'subject']].groupby('subject').mean(),
                     all_results[mask & (all_results['methods']=={'TFD','TG'})][[metric, 'subject']].groupby('subject').mean()))

      # ax.plot([1,2], x.T, 'X-' if ipsiside else 'o-', c='k',
      #         markersize=4*ms, linewidth=2*lw, alpha=0.7,
      #         markeredgecolor='k', fillstyle=fill_style, **filled_marker_style)
      ax.plot([1,2], x.T, markerfacecolor = 'k' if ipsiside else 'darkgrey',
              fillstyle=fill_styles[side], **fancy_marker_style)

  plt.xticks([1,2],['tractfinder \nwithout deformation', 'tractfinder \nwith deformation'], rotation=20)
  ax.set_xlim([0.5,2.5])
  ax.set_ylabel(METRIC_PARAMS[metric]['title'])
  ax.set_ylim([0.18, 0.8])

  l1, = ax.plot(100,100, **single_marker_style, markerfacecolor = '#C3C3C3')
  l2, = ax.plot(100,100, **fancy_marker_style, markerfacecolor = 'k', fillstyle='full')
  l3, = ax.plot(100,100, **single_marker_style, markerfacecolor = 'white')
  l4, = ax.plot(100,100, **fancy_marker_style, markerfacecolor = 'darkgrey', fillstyle='full')
  # ax.legend(l, ['ipsilateral', 'contralateral'], loc='lower center')
  l = ax.legend([(l1, l2), (l3, l4)], ['ipsilateral', 'contralateral'], loc='lower center',
               handler_map={tuple: HandlerTuple(ndivide=None)}, markerscale=1)
  l.get_frame().set_linewidth(lw)
  change_plot_fig.savefig(path.join(results_dir, f'{filename}.pdf'),
              transparent=False, dpi=120)


if __name__ == '__main__':
  main()
