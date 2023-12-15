import numpy as np
import matplotlib.pyplot as plt
from os import path, getlogin
import nibabel as nib
import scipy.ndimage as ndi
from scipy.interpolate import make_interp_spline
import sys
# plt.rcParams['text.usetex'] = True


dirname = f"/Users/{getlogin()}/Documents/{'UCL/CDT' if getlogin() == 'fiona' else 'Research'}/tractfinder/images_and_data/"
img = nib.load(path.join(dirname,'fy','t1_crop.nii.gz'))

#print(img.get_fdata().shape)
#nib.viewers.OrthoSlicer3D(img.get_fdata()).show()

x = 77

imslice = img.get_fdata()[x,:,:]
kspace = np.fft.fftshift(np.fft.ifft2(imslice))
kx = np.fft.fftshift(np.fft.fftfreq(imslice.shape[0]))
ky = np.fft.fftshift(np.fft.fftfreq(imslice.shape[1]))

kmax = max(kx)

# kslice = kspace.real[x]
kx_max_inset = 0.07
ky_max_inset = 0.03
n_ky_lines = np.count_nonzero(abs(ky)<ky_max_inset)
kspace[np.unravel_index(abs(kspace).argmax(), kspace.shape)] *= 0.3
# X = np.arange(kspace.shape[0])
# X = kx
X = kx[np.nonzero(abs(kx)<kx_max_inset)]
fig, ax = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2.5, 2, 1, 1]}, constrained_layout=True)
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

  if j == (n_ky_lines//2 - 1):
    i = 3
    td = np.linspace(X[i], X[i+1], 10)
    ymax = 2*max(Y_)

    ax[0].plot(X_, Y_, 'k-', linewidth=0.5)
    ax[0].plot(X, trail, 'ro', markersize=2)
    # ax[0].plot(, Spline(td), min(Spline(td))-5, color='k', alpha=0.2)
    ax[0].plot([X[i], X[i]], [trail[i], ymax], 'k:',
               [X[i+1], X[i+1]], [trail[i+1], ymax], 'k:', linewidth = 0.5)
    ax[0].plot([X[i], X[i+1]], [ymax, ymax], 'k-|',linewidth=0.5)
    # ax[0].arrow(X[i], ymax, X[i+1]-X[i], 0, length_includes_head=True, width=0.0001, head_width=0.01, head_starts_at_zero=True)
    # ax[0].annotate('$\delta_t$', (X[i] + 0.25*(X[i+1]-X[i]), ymax))
    t = ax[0].text(X[i] + 0.5*(X[i+1]-X[i]), ymax, '$\delta_t$', va='center', ha='center')
    t.set_bbox(dict(facecolor='w', edgecolor='None', pad=0))
    # ax[0].annotate('', xy=(X[i], ymax), xytext=(X[i+1]-X[i], ymax), arrowprops=dict(arrowstyle='<->'))

    ax[0].set_aspect(0.0009)
    # ax[0].set_frame_on(False)
    ax[0].spines[['top','right','left']].set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_xlabel('$k_x$')

    d = 1
    kwargs = dict(marker=[(-1, -d), (1, d)], transform=ax[0].transAxes,
                          color='k', linestyle='None', clip_on=False)
    ax[0].plot([-0.03, 0, 1, 1.03], [0, 0, 0, 0], **kwargs)
    # ax[0].set_ylim(-50,50)
    # ax[0].axis('off')

  m = 3
  ax[1].plot(X_+Y/2 , m*yind+Y_, 'k-', linewidth=0.4, zorder=2*abs(j-kspace.shape[1])+1)
  ax[1].plot(X+Y/2 , m*yind+trail, 'ro', markersize=0.3, zorder=2*abs(j-kspace.shape[1])+1.5)
  ax[1].fill_between(X_+Y/2, m*yind+Y_, min(m*yind+Y_), color='w', alpha=0.9, zorder=2*abs(j-kspace.shape[1]))
  ax[1].axis('off')
  ax[1].set_aspect(0.0015)

  ## 3D attempt
  # ax.plot(X_ , Y_, 'k-', zs=Y, linewidth=0.3, zorder=2*abs(j-kspace.shape[1])+1, zdir='y')
  # ax.add_collection3d(plt.fill_between(X_ , Y_, min(Y_), color='w', alpha=0.9, zorder=2*abs(j-kspace.shape[1])), zs=Y, zdir='y')

zoom=0.5
s = len(kspace)
crop = int((s*(1-zoom))//2)
ax[2].imshow(ndi.rotate(np.absolute(kspace),90)[crop:(s-crop),crop:(s-crop)],
                        extent=[-zoom*kmax, zoom*kmax, -zoom*kmax, zoom*kmax],
                        vmin=0, vmax=4, cmap='gray')
ax[2].add_patch(plt.Rectangle((-kx_max_inset, -ky_max_inset), 2*kx_max_inset, 2*ky_max_inset, ls="--", ec="w", fc="none"))
ax[2].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
ax[2].set_xlabel('$k_x$')
ax[2].set_ylabel('$k_y$')

ax[3].imshow(ndi.rotate(imslice, 90),  cmap='gray', vmin=0, vmax=500)
ax[3].axis('off')

# ax[2].imshow(ndi.rotate(np.absolute(np.fft.fft2(kspace)), 90), cmap='gray')
# plt.colorbar(ax=ax[1])
# plt.tight_layout()
plt.show()
