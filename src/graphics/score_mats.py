import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm, gridspec, rc
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


METHODS_ = ['TF', 'TG', 'AT', 'TSD', 'TSX', 'TGR']
TRACTS = ['af', 'cst', 'ifo', 'or']
HEMS = ['l', 'r']
NAMES = {'tractoinferno':'TractoInferno', 'hcp':'HCP', 'clinical':'Clinical'}

def main():
  current_dir = path.dirname(__file__)
  data_dir = path.join(current_dir, 'raw_results')
  results_dir = path.join(current_dir, pardir, 'figs', 'chapter_5')
  figname = 'score_mats'

  rc('axes', edgecolor='k', linewidth=0.2)
  plt.rcParams.update({'ytick.major.width':0.2, 'xtick.major.width':0.2,
                       'ytick.major.size':2, 'xtick.major.size':2})

  # Only change this for checking no systematic difference based on preprocessing or hospital/cohort!
  split_on = 'tract' #'hem' # 'hem' or 'set' or 'op' or 'ipsi'
  # left_right =  HEMS# HEMS or ['gosh_imri', 'imri'] or [0,1] or [True, False]
  splits = TRACTS
  cfrac = 3

  datas = ['tractoinferno','hcp','clinical']
  n_d = len(datas)
  N = len(METHODS_)
  figcolours = iter(['b','r','g'])

  supfig = plt.figure(figsize=set_size(ratio=3, fraction=0.55))
  subfigs = supfig.subfigures(nrows=3, ncols=1, height_ratios=[N,N-1,N])#, figsize=set_size(ratio=1, fraction=1))

  for d, fig in zip(datas, subfigs):
    METHODS=METHODS_.copy()
    if d == 'tractoinferno':
      datasets = ['tractoinferno_ifo_redo', 'tractoinferno']
    else:
      datasets = [f'{d}_ifo_only', d]
      METHODS.remove('TGR')

    all_results = pd.DataFrame()
    for dataset in datasets:
      datafile = f'{dataset}.pkl'
      all_results = pd.concat((all_results, pd.read_pickle(path.join(data_dir, datafile))))

    n = len(METHODS)
    b = np.zeros((n,n),dtype=int)
    b[np.triu_indices(n,1)[::-1]]=np.arange(0,n*(n-1)/2)
    b[np.triu_indices(n,1)]=np.arange(0,n*(n-1)/2)
    lut = {m1 : { m2 : b[METHODS.index(m1),METHODS.index(m2)] for m2 in METHODS if m2!=m1 } for m1 in METHODS }
    lut_ = [None] * int(factorial(n) / factorial(2) / factorial(n - 2))
    for m1 in lut:
      for m2 in lut[m1]:
        if not lut_[lut[m1][m2]]:
          lut_[lut[m1][m2]] = [m1, m2]

    # fig = plt.figure(figsize=set_size(ratio=1, fraction=1))

    # gs = plt.GridSpec(len(METHODS), len(METHODS))
    # gs = fig.add_gridspec( N+1, N+1, width_ratios=(N-1)*[1]+2*[0.5], height_ratios=(N-1)*[1]+2*[0.5])
    if d == 'tractoinferno':#n < N:
      gs = fig.add_gridspec(N, N)#, width_ratios=(N-1)*[1]+2*[0.5], height_ratios=(N-1)*[1]+2*[0.5])
      # gs_ = gs[:,:].subgridspec(n, n, hspace=0, wspace=0)
    elif d == 'hcp':
      gs = fig.add_gridspec( n, n+cfrac, width_ratios=(n)*[1]+cfrac*[1/cfrac])#, height_ratios=(N-1)*[1]+2*[0.5])
      # gs_ = gs[:n,:n].subgridspec(n, n, hspace=0, wspace=0)
    elif d == 'clinical':
      gs = fig.add_gridspec( n+cfrac, n+cfrac,  height_ratios=(n)*[1]+cfrac*[1/cfrac], width_ratios=(n)*[1]+cfrac*[1/cfrac])#gs_ = gs[:-2,:-2].subgridspec(n, n, hspace=0, wspace=0)
      # gs_ = gs[:,:].subgridspec(n, n, hspace=0, wspace=0)
    # gs_ = gridspec.GridSpecFromSubplotSpec(n,n, subplot_spec=gs[0], hspace=0, wspace=0)
    gs_ = gs[:n,:n].subgridspec(n, n, hspace=0, wspace=0)
    ax = gs_.subplots(sharex=True, sharey=True, subplot_kw={'box_aspect':1})
    # fig.set_facecolor(next(figcolours))

    inds1 = np.triu_indices(n,1)
    # Adjust props
    boxprops['alpha'] = 1
    for i, (m1, m2) in enumerate(combinations(METHODS,2)):

        mask = ((all_results['methods'] == {m1, m2} ))# &
               # (all_results['tract'] == tract ) )#&
               # (all_results['hem'] == h))

        for t, tract in enumerate(TRACTS):
            yi = all_results[mask & (all_results[split_on] == tract)]['density_correlation']
            yj = all_results[mask & (all_results[split_on] == tract)]['binary_dice']

            idx = lut[m1][m2]

            for j, y in enumerate((yi, yj)):
                row, col = inds1[j][idx], inds1[0 if j else 1][idx]
                ax_ = ax[row, col]

                ax_.axvspan(t, t+1,
                                  color = cmap_binary(y.mean()) if j else cmap_generalised(y.mean()))

                box = ax_.boxplot(y, positions=[t+0.5], widths=[0.5],
                                  patch_artist=True, showcaps=0,
                                  boxprops=boxprops, flierprops=flierprops,
                                  medianprops=medianprops,
                                  whiskerprops=whiskerprops)

                # If first column or top row
                if row == 0 and d == 'tractoinferno':
                        ax_.set_xlabel(lut_[idx][1])
                        ax_.xaxis.set_label_position('top')

                if col == 0:
                        ax_.set_ylabel(lut_[idx][1])

    if d == 'hcp' or d == 'clinical':
      plt.setp(ax, xlim=[0,len(splits)], ylim=[0,1], yticks=[0.1,0.5,0.9], xticks=[])
      ax[0][0].set_ylabel(lut_[0][0])
      # ax[0][0].set_title(lut_[0][0])
    else:
      plt.setp(ax, xlim=[0,len(splits)], ylim=[0,1], yticks=[0.1,0.5,0.9], xticks=[])
      ax[0][0].set_ylabel(lut_[0][0])
      ax[0][0].set_xlabel(lut_[0][0])
      ax[0][0].xaxis.set_label_position('top')


    if d == 'hcp':
      # cbar_ax = fig.add_axes([.92, 0.3, 0.05, .577])
      cbar_ax = fig.add_subplot(gs[:n,-cfrac+1])
      cb1 = plt.colorbar(cm.ScalarMappable(norm=None, cmap=cmap_generalised),
                   cax=cbar_ax, label='density correlation', ticks=[0,1])#, aspect=25)
      cb1.set_label('mean density correlation', labelpad=-8)
    if d == 'clinical':
      cbar_ax = fig.add_subplot(gs[-cfrac+1,:n])
      cb2 = plt.colorbar(cm.ScalarMappable(norm=None, cmap=cmap_binary),
                   cax=cbar_ax, orientation='horizontal', ticks=[0,1])
      cb2.set_label('mean Dice similarity coefficient', labelpad=-10)

    # fig.suptitle(NAMES[d], y=.97)
    # fig.subplots_adjust(left=.1, right=.95, top=.9, bottom=.05, wspace=0, hspace=0)


  subfigs[0].suptitle(NAMES['tractoinferno'], y=.95)
  subfigs[0].subplots_adjust(left=.1, right=1, top=.85, bottom=0, wspace=0, hspace=0)

  subfigs[1].suptitle(NAMES['hcp'], y=.97)
  subfigs[1].subplots_adjust(left=.1, right=1, top=.9, bottom=0.05, wspace=0, hspace=0)

  subfigs[2].suptitle(NAMES['clinical'], y=1.02)
  subfigs[2].subplots_adjust(left=.1, right=1, top=.95, bottom=0.1, wspace=0, hspace=0)
  # fig.suptitle(f'All tracts ({", ".join(TRACTS).upper()})')
  supfig.savefig(path.join(results_dir, f'{figname}_group.pdf'),
              transparent=False, dpi=80, bbox_inches="tight")

  plt.cla()



if __name__ == '__main__':
  main()
