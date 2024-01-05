import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
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

  plt.rcParams.update(tex_fonts)
  # plt.rcParams.update({'text.usetex': True, 'svg.fonttype': 'none'})

  filename = 'ts_test_box'
  include_metrics = ['binary_dice','density_correlation','ba_schilling_volume', 'ba_schilling_signed_m1']
  datasets = ['hcp_test']
  # with open(path.join(data_dir, 'ts_tracts.txt')) as f:
  #     TRACTS = [t.lower().rpartition('_')[0] if (('left' in t) or ('right' in t)) else t.lower() for t in f.read().strip().split('\n') ]
  TRACTS = ['af', 'cst', 'ifo', 'or']
  compare_against = 'TG'
  order = METHODS.copy()
  order.remove(compare_against)

  all_results = pd.DataFrame()
  for dataset in datasets:
    datafile = f'{dataset}.pkl'
    all_results = pd.concat((all_results, pd.read_pickle(path.join(data_dir, datafile))))

  n = len(METHODS)
  n_m = len(include_metrics) # nr of metrics
  n_c = n-1 # nr of comparisons
  n_t = len(TRACTS) # nr of tracts

  METRIC_PARAMS['ba_schilling_signed_m1']['lims'] = [-10,10]
  METRIC_PARAMS['ba_schilling_volume']['lims'] = [0,6]

  ## All metrics box plots, grouped by tract
  all_metrics_fig, all_metrics_axs = plt.subplots(nrows=2,
                                                  ncols=ceil(len(include_metrics)/2),
                                                  layout='constrained',
                                                  figsize=set_size(subplots=(2,2)))

  for metric, ax in zip(include_metrics, all_metrics_axs.flat):

      x = np.arange(1, (n_c+1)*n_t, (n_c+1))
      one_of_each = []

      if metric == 'ba_schilling_signed_m1':
          ax.axhline(y=0, color='#D2D2D2', linestyle=':', linewidth=0.5)

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
              # for t in TRACTS:
              #   all_results[defmask & (all_results['tract']==t)].plot(kind='scatter', ax=ax, x=x+i, y=metric)

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


      ax.set_xticks(x+1)
      ax.set_xticklabels(['IFOF' if t.get_text()=='ifo' else t.get_text().upper() for t in ax.get_xticklabels()])
      units = METRIC_PARAMS[metric].get('units')
      ax.set_ylabel(METRIC_PARAMS[metric]['title']+(f' ({units})' if units else ''))
      ax.set_xlabel('')
      ax.tick_params(axis='x', length=0)
      ax.set_ylim(METRIC_PARAMS[metric]['lims'])
      ax.set_title("")

  all_metrics_axs[0,0].legend(one_of_each, order, ncol=len(order), loc='lower center')
  all_metrics_fig.suptitle("HCP105, TractSeg reference bundles")
  plt.margins(0,0)
  all_metrics_fig.savefig(path.join(results_dir, f'{filename}.pdf'),
              transparent=False, dpi=80)#, bbox_inches='tight', pad_inches=0.05)

  ## PART 2: giant scatter plot

  filename = 'ts_scatter'
  include_metrics = ['binary_dice','density_correlation']
  datasets = ['hcp_test_all_tracts_nc', 'hcp_test_all_tracts_c']
  with open(path.join(data_dir, 'ts_tracts.txt')) as f:
      TRACTS = [t.lower().rpartition('_')[0] if (('left' in t) or ('right' in t)) else t.lower() for t in f.read().strip().split('\n') ]
  all_results = pd.DataFrame()
  for dataset in datasets:
    datafile = f'{dataset}.pkl'
    all_results = pd.concat((all_results, pd.read_pickle(path.join(data_dir, datafile))))

  n = len(METHODS)
  n_m = len(include_metrics) # nr of metrics
  n_c = n-1 # nr of comparisons
  n_t = len(TRACTS) # nr of tracts

  big_fig, bix_fig_ax = plt.subplots(nrows=1,
                                    ncols=len(include_metrics),
                                    figsize=set_size(ratio=1.2),
                                    layout='constrained')

  for metric, ax in zip(include_metrics, bix_fig_ax):

    mask = (all_results['methods'] == {compare_against, 'TF'})
    # print(all_results[mask].columns)
    sorted_index = all_results[mask].groupby(['tract','hem']).mean(numeric_only=True)[metric].sort_values().index
    tick_labels = [' '.join(s).upper().strip('_') for s in sorted_index.tolist()]

    for i, method in enumerate(order):
      mask = (all_results['methods'] == {compare_against, method})
      vals = all_results[mask].groupby(['tract','hem']).mean(numeric_only=True)[metric][sorted_index]

      ax.plot(vals, tick_labels, METHOD_SHAPES[method], color=METHOD_PROPS[method]['color'], label=method, ms=4)

      ax.minorticks_on()
      ax.grid(which="major", axis='y', color='#D2D2D2', linestyle=':')
      ax.yaxis.tick_right()
      ax.tick_params(axis='y', which='minor', right=False, left=False)
      ax.set_ylim([-1,len(TRACTS)])
      ax.grid(which="minor", axis='x', color='#D2D2D2', linestyle='-')
      ax.grid(which="major", axis='x', color='#D2D2D2', linestyle='-', lw=1.8)
      ax.set_xlim([0.1, 1])
      ax.set_xlabel(METRIC_PARAMS[metric]['title'])

  bix_fig_ax[0].legend()
  # plt.tight_layout()
  plt.margins(0,0)
  big_fig.savefig(path.join(results_dir, f'{filename}.pdf'),
              transparent=False, dpi=120)#, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
  main()
