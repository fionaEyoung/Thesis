import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from os import path, pardir

S = 0
T = 5
B = 15
P = np.arange(0,25,0.2)

def k(x, l):
  c = np.exp(-l) / (np.exp(-l)-1)
  return (1-c) * np.exp(-l*abs((x)/(B-S))) + c

plt.rcParams['text.usetex'] = True

l = 8
lmax = lambertw(-((B-S)/(T-S))*np.exp(-((B-S)/(T-S))), k=0).real + (B-S)/(T-S)

k_ = k(P-S, l)
k_lmax = k(P-S, lmax)

lin = 1 - abs((P-S)/(B-S))
lin_maxl = 1 - abs((P-S)/(T-S))

y = [0,100]

fig, ax = plt.subplots(figsize=(5,3), layout='tight')

plotwidth=0.8
ax.plot(P-S,k_,linewidth=plotwidth, label=f'$k_{{l={l:d}}}$')
ax.plot(P-S,k_lmax,linewidth=plotwidth, label=f'$k_{{l=l_{{max}}}}$')
ax.plot(P-S,lin,linewidth=plotwidth, label=r'$k=1-\frac{{D_P}}{{D_b}}$')
ax.plot(P-S,lin_maxl,linewidth=plotwidth, label=r'$k=1-\frac{{D_P}}{{D_t}}$')

ax.plot([S+B,S+B], y, 'k--', [S-B,S-B], y, 'k--', [S+T,S+T], y, 'k:', [S-T,S-T], y, 'k:' )

ax.set_ylim([0,1.1])
ax.set_yticks([])
ax.set_xlim([0,B+1])
ax.set_xticks([T,B])
ax.set_xticklabels(['$D_t$', '$D_b$'])
ax.set_ylabel('$k$', fontsize=15)
ax.set_xlabel(r'$D_p$', fontsize=15)

leg = plt.legend(loc='upper right', fancybox=False, edgecolor='k', framealpha=1)
# leg.legendPatch.set(linestyle='--')

plt.savefig(path.join(path.dirname(__file__), 'draft_figs', 'chapter_3', 'k.png'), dpi=160)
