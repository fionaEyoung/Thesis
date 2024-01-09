import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects, patches
from matplotlib.path import Path
from os import path, getlogin, pardir
import nibabel as nib
import scipy.ndimage as ndi
from scipy.interpolate import make_interp_spline
import sys
from figutils import *

# plt.rcParams['text.usetex'] = True

current_dir = path.dirname(__file__)
results_dir = path.join(current_dir, pardir, 'figs', 'chapter_1')
filename = 'kspace'

dirname = f"/Users/{getlogin()}/Documents/{'UCL/CDT' if getlogin() == 'fiona' else 'Research'}/tractfinder/images_and_data/"
img = nib.load(path.join(dirname,'fy','t1_crop.nii.gz'))

#print(img.get_fdata().shape)
#nib.viewers.OrthoSlicer3D(img.get_fdata()).show()

x = 77
z = 127

# imslice = img.get_fdata()[x,:,:]
imslice = img.get_fdata()[:,:,z]
kspace = np.fft.fftshift(np.fft.ifft2(imslice))
kx = np.fft.fftshift(np.fft.fftfreq(imslice.shape[0]))
ky = np.fft.fftshift(np.fft.fftfreq(imslice.shape[1]))

kmax_x = max(kx)
kmax_y = max(ky)

# kslice = kspace.real[x]
kx_max_inset = 0.07
ky_max_inset = 0.03
n_ky_lines = np.count_nonzero(abs(ky)<ky_max_inset)
kspace[np.unravel_index(abs(kspace).argmax(), kspace.shape)] *= 0.2
# X = np.arange(kspace.shape[0])
# X = kx
X = kx[np.nonzero(abs(kx)<kx_max_inset)]
# fig, axs = plt.subplots(2, 3,
#                         gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios':[1,2]},
#                         constrained_layout=True)
fig = plt.figure(figsize=set_size(ratio=1/2.11, fraction=1.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1,2.5], wspace=0.05)
ax0 = fig.add_subplot(gs[0,:])
ax_k = fig.add_subplot(gs[1,0])
ax_kimg = fig.add_subplot(gs[1,1])
ax_img = fig.add_subplot(gs[1,2])
ymin = 1000
# ax = plt.figure().add_subplot(projection='3d')
# ax[0].plot(kspace[x], 'k.')
for j, Y in reversed(list(enumerate(ky[np.nonzero(abs(ky)<ky_max_inset)]))): #np.arange(kspace.shape[1])[::-1]:
  yind=np.flatnonzero(abs(ky)<ky_max_inset)[0]+j
  # sys.exit(0
  # trail = kspace.real[np.nonzero(abs(kx)<kmax_inset),yind].flat
  trail = kspace.real[np.nonzero(abs(kx)<kx_max_inset),yind].flat
  Spline = make_interp_spline(X, trail)

  X_ = np.linspace(X.min(), X.max(), 600)
  Y_ = Spline(X_)

  # Single kspace line
  # if j == (n_ky_lines//2 - 1):
  if j==0:
    i = 3
    td = np.linspace(X[i], X[i+1], 10)
    ymax = 2*max(Y_)

    ax = ax0
    ax.plot(X_, Y_, 'k-', linewidth=0.5)
    ax.plot(X, trail, 'ro', markersize=2)
    # ax.plot(, Spline(td), min(Spline(td))-5, color='k', alpha=0.2)
    ax.plot([X[i], X[i]], [trail[i], ymax], 'k:',
               [X[i+1], X[i+1]], [trail[i+1], ymax], 'k:', linewidth=0.5)
    ax.plot([X[i], X[i+1]], [ymax, ymax], 'k-|',linewidth=lw)
    # ax.arrow(X[i], ymax, X[i+1]-X[i], 0, length_includes_head=True, width=0.0001, head_width=0.01, head_starts_at_zero=True)
    # ax.annotate('$\delta_t$', (X[i] + 0.25*(X[i+1]-X[i]), ymax))
    t = ax.text(X[i] + 0.5*(X[i+1]-X[i]), ymax, '$t_d$', va='center', ha='center')
    t.set_bbox(dict(facecolor='w', edgecolor='None', pad=0))
    # ax.annotate('', xy=(X[i], ymax), xytext=(X[i+1]-X[i], ymax), arrowprops=dict(arrowstyle='<->'))

    # ax.set_aspect(0.0009)
    # ax.set_frame_on(False)
    ax.spines[['top','right','left']].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xticks([])
    # ax.set_xlabel('$k_x$')

    d = 1 #slantyness
    off = 0.002
    kwargs = dict(marker=[(-1, -d), (1, d)], transform=ax.transAxes,
                          color='k', linestyle='None', clip_on=False, mew=lw)
    ax.plot([-0.01-off, 0-off, 1, 1.01], [0, 0, 0, 0], **kwargs)
    # ax[0].set_ylim(-50,50)
    # ax[0].axis('off')

  m = 3
  ax = ax_k
  ylim = [300,345]
  ymin = min(ymin, min(m*yind+Y_))
  ax.plot(X_+Y/2 , m*yind+Y_, 'k-', linewidth=0.4, zorder=2*abs(j-kspace.shape[1])+1, clip_on=False)
  ax.plot(X+Y/2 , m*yind+trail, 'ro', markersize=0.5, fillstyle='full', zorder=2*abs(j-kspace.shape[1])+1.5, clip_on=False)
  ax.fill_between(X_+Y/2, m*yind+Y_, ymin, color='w', alpha=0.8, zorder=2*abs(j-kspace.shape[1]), clip_on=False)
  ax.axis('off')
  ax.set_aspect(0.0021)
  ax.set_ylim(ylim)
  ax.set_xlim([-0.8*kx_max_inset,1.1*kx_max_inset])

  ## 3D attempt
  # ax.plot(X_ , Y_, 'k-', zs=Y, linewidth=0.3, zorder=2*abs(j-kspace.shape[1])+1, zdir='y')
  # ax.add_collection3d(plt.fill_between(X_ , Y_, min(Y_), color='w', alpha=0.9, zorder=2*abs(j-kspace.shape[1])), zs=Y, zdir='y')

zoom=0.6
# zoom=1
s_x, s_y = kspace.shape
crop_x, crop_y = ((np.array(kspace.shape)*(1-zoom))//2).astype(int)
ax = ax_kimg
ax.imshow(ndi.rotate(np.absolute(kspace),90)[crop_y:(s_y-crop_y),crop_x:(s_x-crop_x)],
                        extent=[-zoom*kmax_x, zoom*kmax_x, -zoom*kmax_y, zoom*kmax_y],
                        vmin=0, vmax=4, cmap='gray')
# Inset box
ax.add_patch(patches.PathPatch(Path([[-kx_max_inset, -ky_max_inset],
                                     [-kx_max_inset, ky_max_inset],
                                     [kx_max_inset, ky_max_inset],
                                     [kx_max_inset, -ky_max_inset],
                                     [-kx_max_inset, -ky_max_inset]]),
             ls="--", ec="w", fc="none",
             path_effects=[patheffects.SimpleLineShadow(offset=(.5, -.5), alpha=.5), patheffects.Normal()]))
# Zoom lines
ax.plot([-zoom*kmax_x,-kx_max_inset], [0.6*zoom*kmax_y, ky_max_inset], 'w-', lw=lw)
ax.plot([-zoom*kmax_x,-kx_max_inset], [-0.75*zoom*kmax_y, -ky_max_inset], 'w-', lw=lw)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
# ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.xaxis.set_label_position("top")
ax.set_xlabel('$k_x$')

ax = ax_img
ax.imshow(ndi.rotate(imslice, 90),  cmap='gray', vmin=0, vmax=500)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
ax.set_xlabel('$x$')
ax.yaxis.set_label_position("right")
ax.set_ylabel('$y$')

# ax[2].imshow(ndi.rotate(np.absolute(np.fft.fft2(kspace)), 90), cmap='gray')
# plt.colorbar(ax=ax[1])
# plt.tight_layout()
# plt.show()
plt.margins(0,0)
fig.savefig(path.join(results_dir, f'{filename}.pdf'),
            transparent=False, dpi=300, bbox_inches="tight")
