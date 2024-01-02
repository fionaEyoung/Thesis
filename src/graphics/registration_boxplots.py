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


METHODS = ['TF', 'TFA', 'TG']
TRACTS = ['af', 'cst', 'ifo', 'or']
HEMS = ['l', 'r']

def main():
  current_dir = path.dirname(__file__)
  data_dir = path.join(current_dir, 'raw_results')
  results_dir = path.join(current_dir, pardir, 'figs', 'chapter_4')
  figname = 'registration_2'

  # plt.rcParams.update({'font.size': 18})
  plt.rcParams.update(tex_fonts)
  plt.rcParams.update({'hatch.linewidth':boxlw})

  include_metrics = ['binary_dice','density_correlation']#,'ba_schilling_volume', 'ba_schilling_signed_m1']
  datasets = ['tractoinferno_compare_nrr', 'tractoinferno_ifo_redo_compare_nrr']
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

  ## All metrics box plots, grouped by tract
  all_metrics_fig, all_metrics_axs = plt.subplots(ncols=2,
                                                  nrows=ceil(len(include_metrics)/2),
                                                  layout='constrained',
                                                  figsize=set_size(subplots=(ceil(len(include_metrics)/2),2)))

  for metric, ax in zip(include_metrics, all_metrics_axs.flat):

      x = np.arange(1, (n_c+1)*n_t, (n_c+1))
      one_of_each = []

      for i, method in enumerate(order):
          # Left and right all combined
          mask = (all_results['methods'] == {compare_against, method})
          if not metric == 'ba_schilling_signed_m1':

              box = all_results[mask].boxplot(column=metric, by='tract', ax=ax, grid=False,
                                        return_type='dict', patch_artist=True,
                                        flierprops=flierprops, showcaps=0,
                                        # boxprops=dict(hatch='////'),
                                        boxprops=boxprops, whiskerprops=boxlineprops, medianprops=boxlineprops,
                                        positions=x+i)
              # set_box_colours(box[metric], color=METHOD_PROPS[method]['color'])
              set_box_colours(box[metric], hatch=METHOD_PROPS[method]['hatch'], color='k', facecolor='w')
              one_of_each.append(box[metric]['boxes'][0])
              # for t in TRACTS:
              #   all_results[defmask & (all_results['tract']==t)].plot(kind='scatter', ax=ax, x=x+i, y=metric)

          else:
              # special case for signed
              box = all_results[mask].assign(
                  signed=lambda df: np.where(df.method1.apply(lambda x: x['name']) == method, df.ba_schilling_signed_m1, df.ba_schilling_signed_m2)
              ).boxplot(column='signed', by='tract', ax=ax, grid=False,
                        return_type='dict', patch_artist=True,
                        flierprops=flierprops,  showcaps=0,
                        # boxprops=dict(alpha=0.2),
                        positions=x+i)
              # set_box_colours(box['signed'], color=METHOD_PROPS[method]['color'])
              set_box_colours(box['signed'], hatch=METHOD_PROPS[method]['hatch'], color='k', facecolor='w')
              one_of_each.append(box['signed']['boxes'][0])

      if metric == 'ba_schilling_signed_m1':
          ax.axhline(y=0, c='#8a8a8a', ls='--')

      ax.set_xticks(x+1)
      ax.set_xticklabels(['IFOF' if t.get_text()=='ifo' else t.get_text().upper() for t in ax.get_xticklabels()])
      units = METRIC_PARAMS[metric].get('units')
      ax.set_ylabel(METRIC_PARAMS[metric]['title']+(f' ({units})' if units else ''))
      ax.set_xlabel('')
      ax.tick_params(axis='x', length=0)
      ax.set_ylim(METRIC_PARAMS[metric]['lims'])
      ax.set_title("")

  all_metrics_axs[0].legend(one_of_each, [METHOD_PROPS[o].get('name', o) for o in order], ncol=len(order),
          columnspacing=1, markerscale=0.5, handlelength=1)
  all_metrics_fig.suptitle("")
  plt.margins(0,0)
  all_metrics_fig.savefig(path.join(results_dir, f'{figname}.pdf'),
              transparent=False, dpi=120)#, bbox_inches="tight")




if __name__ == '__main__':
  main()
