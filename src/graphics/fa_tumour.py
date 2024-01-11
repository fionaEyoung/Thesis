import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm, rc
from matplotlib.ticker import ScalarFormatter
from os import path, getlogin, pardir
from figutils import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.image import imread

def main():
  current_dir = path.dirname(__file__)
  data_dir = path.join(current_dir, 'raw_results')
  results_dir = path.join(current_dir, pardir, 'figs', 'chapter_4')

  filename = 'tumour_fa'
  fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True,
                          figsize=set_size(subplots=(2,3)))

  c_tumour = METHOD_PROPS['TF']['color']
  c_brain = METHOD_PROPS['TG']['color']
  nbins = 50
  xlim = [0,1]

  for s, ax in zip(range(1,7), axs.flat):
    brain_fa = np.loadtxt(path.join(data_dir, f'{s}_brain_fa_vals.txt'))
    tumour_fa = np.loadtxt(path.join(data_dir, f'{s}_tumour_fa_vals.txt'))
    picture = imread(path.join(data_dir, f'{s}.png'))

    weights = [np.ones(len(x))/len(x) for x in (brain_fa, tumour_fa)]
    ax.hist([brain_fa, tumour_fa], weights=weights, color=[c_brain, c_tumour],
            bins=nbins, range=xlim, alpha=0.8, histtype='stepfilled',
            label=['Whole brain', 'Tumour only'])

    ax.set_ylim([0,0.3])
    ax.set_yticks(np.arange(0,0.4,0.1))
    ax.set_title(f'Subject {s}')
    plt.setp(ax.spines.values(), linewidth=lw)

    ax_ = inset_axes(ax, width='50%', height='80%')
    ax_.imshow(picture)
    ax_.axis('off')

  axs[1,1].set_xlabel('Fractional anisotropy')
  axs[0,1].legend(loc='upper left')
  for ax in axs[:,1:].flat:
    ax.tick_params(axis='y', length=0)
  for ax in axs.flat:
    ax.tick_params(width=lw)

  fig.subplots_adjust(wspace=0.08, hspace=0.2)
  plt.margins(0,0)
  set_ax_size(ax=axs, ratio=0.45)
  fig.savefig(path.join(results_dir, f'{filename}.pdf'),
              transparent=False, dpi=300, bbox_inches="tight", pad_inches=0.01)

if __name__ == '__main__':
  main()
