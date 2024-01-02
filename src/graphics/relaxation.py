import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
# plt.rcParams['text.usetex'] = True
plt.rcParams.update ({'text.usetex': True, "svg.fonttype": 'none'})

n = 1000
s = 5
g = '#6e6e6e'

t = np.linspace(0, 7*np.pi, n);
r = np.linspace(s*1.5, 0, n);
mz = np.linspace(0, s*2, n);

# Spiral
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
ax.plot(r*np.sin(t), r*np.cos(t), mz, linewidth=0.5, color=g);

xyz = 2*s*np.array([[[-1, 1],[0, 0], [0, 0]],
                   [ [0, 0], [-1, 1],[0, 0]],
                   [ [0, 0], [0, 0], [0, 1.3]]])
L = [ ax.plot(*xyz[0]),
      ax.plot(*xyz[1]),
      ax.plot(*xyz[2])]
plt.setp(L,  linewidth=0.5, color='k')

# M
# quiver version
# i = 450
# ax.quiver(0, 0, 0, r[i]*np.sin(t[i]), r[i]*np.cos(t[i]), mz[i],
#           linewidth=2, arrow_length_ratio=0.1, color='k');
# # Ortho keyvals
# props = dict(linewidth=2, arrow_length_ratio=0.1, color=g)
# # Mxy
# mxy = np.array([r[i]*np.sin(t[i]), r[i]*np.cos(t[i]), 0])
# ax.quiver(0, 0, 0, *mxy, **props);
# ax.text(*mxy/2, "$M_{xy}$")
# # Mz
# ax.quiver(0, 0, 0, 0, 0, mz[i], **props);

# M
# line version
i = 450
ax.plot([0, r[i]*np.sin(t[i])], [0, r[i]*np.cos(t[i])], [0, mz[i]],
          linewidth=2, color='k');
ax.text(r[i]*np.sin(t[i])*0.7, r[i]*np.cos(t[i])*0.7, mz[i]*0.9, r"\$\mathbf{M}$")

# Ortho keyvals
props = dict(linewidth=2, color=g) #,arrow_length_ratio=0.1, color=g)
# Mxy
mxy = np.array([[0,r[i]*np.sin(t[i])], [0,r[i]*np.cos(t[i])], [0,0]])
ax.plot(*mxy, **props);
# rec = Rectangle((0,0), mxy[0,1], mxy[1,1], zorder=0, ec=None, fc=g, alpha=0.5)
# ax.add_patch(rec)
# art3d.pathpatch_2d_to_3d(rec, z=0, zdir="z")
ax.text(*((mxy[:,1]-[s*0.2,0,0])), "\$M_{xy}$")

# Mz
ax.plot([0,0], [0,0], [0,mz[i]], **props);
ax.text(0, s*0.2, mz[i]*0.9, "\$M_{z}$")

## Visualisation
ax.view_init(elev=32, azim=-50)
ax.set_aspect('equal')
ax.autoscale(enable=True, tight=True)
ax.axis('off')
plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout()
# plt.show()
plt.savefig('figs/chapter_1/relaxation.svg', format='svg')
plt.savefig('figs/chapter_1/relaxation.pdf', format='pdf')
