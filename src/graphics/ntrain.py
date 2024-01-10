import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.ticker import ScalarFormatter
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
TRACTS = ['af', 'cst', 'ifo', 'or']
NTRAIN = ['1', '3', '5', '10', '15', '30', '63']

def rand_jitter(arr, stdev=None):
    if stdev is None:
        # https://stackoverflow.com/a/21276920
        stdev = .005 * (max(arr) - min(arr))
        if stdev == 0:
            stdev=.02
    return arr + np.random.randn(len(arr)) * stdev


def main():
  no_stats = True;

  current_dir = path.dirname(__file__)
  data_dir = path.join(current_dir, 'raw_results')
  results_dir = path.join(current_dir, pardir, 'figs', 'chapter_3')

  plt.rcParams.update(tex_fonts)
  # plt.rcParams.update({'text.usetex': True, 'svg.fonttype': 'none'})

  filename = 'ntrain'
  include_metrics = ['binary_dice','density_correlation']
  compare_against = 'TG'
  order = METHODS.copy()
  order.remove(compare_against)

  n = len(METHODS)
  n_m = len(include_metrics) # nr of metrics
  n_c = len(NTRAIN) # nr of comparisons
  n_t = len(TRACTS) # nr of tracts
  method = 'TF'

  all_results = []
  for ntrain in ['_'+x if x != '63' else '' for x in NTRAIN ]:
      datafile = f'hcp_test{ntrain}.pkl'
      all_results.append( pd.read_pickle(path.join(data_dir, datafile)) )
  all_results = pd.concat([all_results[i].assign(ntrain=int(NTRAIN[i])) for i in range(n_c)])

  if not no_stats:
    for metric in include_metrics:
        print(f"{' '.join(metric.split('_'))} scores:\n")
        for i, ntrain in enumerate(NTRAIN):
            mask = (all_results[i]['methods'] == {compare_against, method})
            print(f"Number of atlas training subjects: {ntrain}")
            print(all_results[i][mask].groupby('tract')[metric].agg(["mean", "std"]))

  ## Scatter plots


  n_subjects = 42

  linthresh = 16
  metric = 'binary_dice'
  scatter_fig, scatter_ax = plt.subplots()
  scatter_ax.set_xscale("symlog", linthresh=linthresh)
  bins = np.arange(0.5, 0.9, 0.02)

  for t in TRACTS:
  # for t in ['cst']:

      mask = ((all_results['methods'] == {compare_against, method}) & (
              all_results['tract'] == t ))
      mask2 = ((all_results['methods'] == {compare_against, 'TSD'}) & (
              all_results['tract'] == t ))

      # plt.plot(rand_jitter(all_results[mask]['ntrain']), all_results[mask]['binary_dice'], 'o', color=TRACT_COLOURS[t], alpha=0.1)

      for n in NTRAIN:
          data = all_results[(mask & (all_results['ntrain'] == int(n)))]

          inds = np.digitize(data['binary_dice'], bins)
          # stds = np.histogram(data['binary_dice'], bins)[0][inds]*(0.01/(1-((int(n))/70)))
          if int(n)>linthresh:
            stds = np.log10(np.histogram(data['binary_dice'], bins)[0][inds])*(np.log10(int(n))/1.6)
          else:
            stds = np.histogram(data['binary_dice'], bins)[0][inds]*0.013

          plt.plot(rand_jitter(data['ntrain'], stdev=stds), data['binary_dice'],
                                  'o', ms=5, color=TRACT_COLOURS[t], alpha=0.3, mew=0)
          if n == NTRAIN[-1]:
            data_ts = all_results[(mask2 & (all_results['ntrain'] == int(n)))]
            inds_ts = np.digitize(data_ts['binary_dice'], bins)
            stds_ts = np.log10(np.histogram(data_ts['binary_dice'], bins)[0][inds_ts])*np.log10(int(n))

            plt.plot(rand_jitter(data_ts['ntrain']+15, stdev=stds_ts), data_ts['binary_dice'],
                                    '^', ms=5,
                                    color=TRACT_COLOURS[t], alpha=0.3, mew=0)

      # sb.swarmplot(all_results[mask]['binary_dice'], x=all_results[mask]['ntrain'], color=TRACT_COLOURS[t], alpha=0.1)
      # sb.swarmplot(data=all_results[mask], y='binary_dice', x='ntrain', color=TRACT_COLOURS[t], alpha=0.3, native_scale=True)

  for t in TRACTS:
  # for t in ['cst']:
      mask = ((all_results['methods'] == {compare_against, method}) & (
              all_results['tract'] == t ))
      mask2 = ((all_results['methods'] == {compare_against, 'TSD'}) & (
              all_results['tract'] == t ) & (
              all_results['ntrain']== int(NTRAIN[-1])))

      plt.plot([int(x) for x in NTRAIN], all_results[mask].groupby('ntrain')['binary_dice'].mean(),
              'o-', color=TRACT_COLOURS[t], mec='k', mew=0.5,
              label=t.upper())
      plt.plot(int(NTRAIN[-1])+15, all_results[mask2]['binary_dice'].mean(),
              '^-', color=TRACT_COLOURS[t], mec='k', mew=0.5)

      # print("Tract: ", t)
      # print("\ttractfinder: ", all_results[mask].groupby('ntrain')['binary_dice'].mean())
      # print("\tTractSeg: ", all_results[mask2]['binary_dice'].mean())

  # Additionally plot results from Liu et al. 2023
  plt.plot(0.5, 0.719, 's', color=TRACT_COLOURS['cst'], mec='k', mew=0.5)
  plt.plot(0.5, 0.624, 's', color=TRACT_COLOURS['or'], mec='k', mew=0.5)

  plt.plot(100,100, 'o', mec='k', color='w', mew=0.5, label='tractfinder')
  plt.plot(100,100, '^', mec='k', color='w', mew=0.5, label='TractSeg')
  plt.plot(100,100, 's', mec='k', color='w', mew=0.5, label='TractSeg+Oneshot (Liu \\textit{et al.}, 2023)')

  scatter_ax.set_ylabel("Dice similarity coefficient")
  scatter_ax.set_xlabel("Number of training subjects")

  (lines, labels) = plt.gca().get_legend_handles_labels()
  lines.insert(0, plt.Line2D([100],[100], linestyle='none', marker='none'))
  lines.insert(3, plt.Line2D([100],[100], linestyle='none', marker='none'))
  labels.insert(0,'')
  labels.insert(3,'')
  lgd = scatter_ax.legend(lines,labels,loc='lower left', ncol=3, columnspacing=0.9, handletextpad=0.5, markerscale=1)

  ylim = [0.48,0.92]
  scatter_ax.minorticks_on()
  scatter_ax.set_ylim(ylim)
  scatter_ax.yaxis.set_ticks(np.arange(0.5, 1, 0.1))
  scatter_ax.xaxis.set_tick_params(which='minor', bottom=False)
  scatter_ax.grid(which="both", axis='y', color='#D2D2D2', linestyle=':')

  scatter_ax.set_xticks([int(x) for x in NTRAIN])
  scatter_ax.xaxis.set_major_formatter(ScalarFormatter())
  scatter_ax.set_xlim([-0.5,100])

  # scatter_fig[0].legend()
  plt.tight_layout()
  set_ax_size(ax=scatter_ax, ratio=.55)
  plt.margins(0,0)
  scatter_fig.savefig(path.join(results_dir, f'{filename}.pdf'),
              transparent=False, dpi=300, bbox_inches="tight", pad_inches=0.01)


if __name__ == '__main__':
  main()
