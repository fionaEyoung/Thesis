import matplotlib.pyplot as plt
from matplotlib import colormaps, cm

def set_box_colours(bp, index=None, **kwargs):
    # print(bp.keys())

    c = kwargs.get('color', False)
    if c:
      for prop in bp.keys():
          if not index:
              plt.setp(bp[prop], color=c)
          else:
              plt.setp(bp[prop][index], color=c)
    for b in bp['boxes']:
      b.set(**kwargs)

#https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width=452.9679, ratio=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * (ratio or golden_ratio) * (subplots[0] / subplots[1])
    # print(fig_width_in)
    return (fig_width_in, fig_height_in)

# https://stackoverflow.com/a/44971177
def set_ax_size(width=452.9679, ratio=None, fraction=1, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2

    # Width of figure (in pts)
    ax_width_in  = width * fraction * inches_per_pt
    ax_height_in = ax_width_in * (ratio or golden_ratio)

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(ax_width_in)/(r-l)
    figh = float(ax_height_in)/(t-b)
    ax.figure.set_size_inches(figw, figh)

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "sans-serif",
    'font.serif'  : 'cmss',
    # Use 10pt font in plots, to match 10pt font in document
    "font.size": 12,
    "axes.labelsize": 'small',
    "axes.titlesize": 'small',
    "figure.labelsize": 'medium',
    "figure.titlesize": 'medium',
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 'x-small',
    "legend.title_fontsize": "small",
    "xtick.labelsize": 'x-small',
    "ytick.labelsize": 'x-small'
}

lw = 0.5
ms = 1

plot_styles = {
    "legend.framealpha": 1,
    "legend.columnspacing":0.5,
    "legend.handletextpad":0.3,
    "legend.markerscale":0.5,
    "legend.handlelength":0.7,
    "hatch.linewidth":lw
}

plt.rcParams.update(tex_fonts)
plt.rcParams.update(plot_styles)

#https://davidmathlogic.com/colorblind/#%23000000-%233D840B-%23F38B03-%235D32B1
TRACT_COLOURS = {'af' :'#F38B03',
                 'or' :'#3D840B',
                 'cst':'#5D32B1',
                 'ifo':'#CCD616'}
METHOD_PROPS = {'TSX': dict(color='#FF8100', hatch=''),
                  'TSD': dict(color='#C80000', hatch=''),
                  'TG' : dict(color='#7797A7', hatch=''),
                  'TGR': dict(color='#1DAFFF', hatch=''),
                  'TF' : dict(color='#FFC818', hatch='', name='Affine'),
                  'AT' : dict(color='#F046A8', hatch=''),
                  'TFF': dict(color='#F38B03', hatch=''),
                  'TFD': dict(color='#F38B03', hatch=''),
                  'TFA': dict(color='#BECB34', hatch='///////', name='Nonlinear')}
METHOD_SHAPES = {'TSX': 'v',
                  'TSD': '^',
                  'TG' : 's',
                  'TGR': 's',
                  'TF' : 'o',
                  'AT' : 'd',
                  'TFF': 'x',
                  'TFD': 'x',
                  'TFA': 'x'}
METRIC_PARAMS = {'binary_dice': {'lims': [0,1],
                                 'title': 'Dice similarity coefficient'},
               'hd':            {'lims': [0,40],
                                'title': 'Hausdorff distance',
                                'units': 'mm'},
               'ba_schilling_boundary': {'lims': [0,15],
                                'title': 'Bundle adjacency (Boundary)',
                                'units': 'mm'},
               'ba_schilling_volume': {'lims': [0,15],
                                'title': 'Bundle distance',
                                'units': 'mm'},
               'weighted_dice': {'lims': [0,1],
                                 'title': 'Weighted DSC'},
               'density_correlation': {'lims': [0,1],
                                 'title': 'Density correlation'},
               'ba_schilling_signed_m1': {'lims': [-15,15],
                                 'title': 'Signed bundle distance',
                                 'units': 'mm'}}

cmap_generalised = colormaps['inferno']
cmap_binary = colormaps['viridis']
# Boxplot parameters
c = 'w'
boxprops = dict(linestyle='-', color=c, facecolor=c, linewidth=lw, alpha=0.7)
boxlineprops = dict(linestyle='-', color=c, linewidth=lw, alpha=0.7)
# flierprops = dict(marker='.', markerfacecolor=c, markeredgecolor=c, markersize=1)
medianprops = dict(linestyle='-', linewidth=lw, color='k')
whiskerprops = dict(color=c)
flierprops = dict(marker='o', markerfacecolor='k', markeredgecolor='none', markersize=ms, alpha=0.5)
meanpointprops = dict(marker='o', markeredgecolor='k', markersize=5, markerfacecolor='k')
meanlineprops = dict(linestyle='--', linewidth=lw, color='purple')
