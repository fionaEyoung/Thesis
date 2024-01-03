import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from os import path, pardir
from figutils import *

S = 0
T = 4
B = 15
P = np.arange(0,25,0.2)

def k(x, l):
  c = np.exp(-l) / (np.exp(-l)-1)
  return (1-c) * np.exp(-l*abs((x)/(B-S))) + c

plt.rcParams.update(tex_fonts)
current_dir = path.dirname(__file__)
out_dir = path.join(current_dir, pardir, 'figs', 'chapter_4')

l = 8
lmax = lambertw(-((B-S)/(T-S))*np.exp(-((B-S)/(T-S))), k=0).real + (B-S)/(T-S)

k_ = k(P-S, l)
k_lmax = k(P-S, lmax)

lin = 1 - abs((P-S)/(B-S))
lin_maxl = 1 - abs((P-S)/(T-S))

y = [0,100]

fig, ax = plt.subplots(figsize=set_size(fraction=0.65), layout='constrained')

# Indicate tumour and brain boundaries
# ax.plot([S+B,S+B], y, 'k--', [S-B,S-B], y, 'k--', [S+T,S+T], y, 'k:', [S-T,S-T], y, 'k:' )
ax.axvspan(S, T, facecolor='k', alpha=0.1)
ax.axvspan(S, B, facecolor='k', alpha=0.1)

plotwidth=0.6
# ax.plot(P-S,k_,linewidth=plotwidth, label=f'$k_{{l={l:d}}}$')
ax.plot(P-S,k_lmax, 'k-', linewidth=plotwidth, label=f'$k_{{l=l_{{max}}}}$')
ax.plot(P-S,lin_maxl, 'k--', linewidth=plotwidth, label=r'$k=1-\frac{{D_P}}{{D_t}}$')
ax.plot(P-S,lin, 'k:', linewidth=plotwidth, label=r'$k=1-\frac{{D_P}}{{D_b}}$')

ax.set_ylim([0,1.1])
ax.set_yticks([])
ax.set_xlim([0,B+1])
ax.set_xticks([T,B])
ax.set_xticklabels(['$D_t$', '$D_b$'])
ax.set_ylabel('$k$', fontsize=15)
ax.set_xlabel(r'$D_p$', fontsize=15)

leg = plt.legend(loc='upper center',  bbox_to_anchor=(T+(B-T)/2, 1),bbox_transform=ax.transData,
                  frameon=False)
# leg.legendPatch.set(linestyle='--')
plt.margins(0,0)
plt.savefig(path.join(out_dir, 'k.pdf'), dpi=300)
