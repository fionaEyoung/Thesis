import numpy as np
import matplotlib.pyplot as plt
from os import path
import nibabel as nib
import scipy.ndimage as ndi
from scipy.interpolate import make_interp_spline
import sys
# plt.rcParams['text.usetex'] = True

dirname = "/Users/fionayoung/Documents/Research/tractfinder/images_and_data/"
img = nib.load(path.join(dirname,'fy','t1_crop.nii.gz'))

#print(img.get_fdata().shape)
#nib.viewers.OrthoSlicer3D(img.get_fdata()).show()

x = 77

imslice = img.get_fdata()[x,:,:]
kspace = np.fft.fftshift(np.fft.ifft2(imslice))
kx = np.fft.fftshift(np.fft.fftfreq(imslice.shape[0]))
ky = np.fft.fftshift(np.fft.fftfreq(imslice.shape[1]))

# kslice = kspace.real[x]
kmax_inset = 0.1
kmax = max(kx)
# X = np.arange(kspace.shape[0])
# X = kx
X = kx[np.nonzero(abs(kx)<kmax_inset)]
fig, ax = plt.subplots(ncols=3)
# ax = plt.figure().add_subplot(projection='3d')

# ax[0].plot(kspace[x], 'k.')
for j, Y in reversed(list(enumerate(ky[np.nonzero(abs(ky)<kmax_inset)]))): #np.arange(kspace.shape[1])[::-1]:
  yind=np.flatnonzero(abs(ky)<kmax_inset)[0]+j
  # sys.exit(0
  trail = kspace.real[np.nonzero(abs(kx)<kmax_inset),yind].flat
  Spline = make_interp_spline(X, trail)

  X_ = np.linspace(X.min(), X.max(), 600)
  Y_ = Spline(X_)

  ax[0].plot(X_+Y/2 , 2*yind+Y_, 'k-', linewidth=0.3, zorder=2*abs(j-kspace.shape[1])+1)
  ax[0].fill_between(X_+Y/2, 2*yind+Y_, min(yind+Y_), color='w', alpha=0.9, zorder=2*abs(j-kspace.shape[1]))
  ax[0].axis('off')

  ## 3D attempt
  # ax.plot(X_ , Y_, 'k-', zs=Y, linewidth=0.3, zorder=2*abs(j-kspace.shape[1])+1, zdir='y')
  # ax.add_collection3d(plt.fill_between(X_ , Y_, min(Y_), color='w', alpha=0.9, zorder=2*abs(j-kspace.shape[1])), zs=Y, zdir='y')

ax[1].imshow(ndi.rotate(np.absolute(kspace),90),
                        extent=[-kmax, kmax, -kmax, kmax],
                        vmin=0, vmax=4, cmap='gray')
ax[1].add_patch(plt.Rectangle((-kmax_inset, -kmax_inset), 2*kmax_inset, 2*kmax_inset, ls="--", ec="w", fc="none"))
ax[1].set_xlabel('$k_x$')
ax[1].set_ylabel('$k_y$')

ax[2].imshow(ndi.rotate(imslice, 90),  cmap='gray')
ax[2].axis('off')

# ax[2].imshow(ndi.rotate(np.absolute(np.fft.fft2(kspace)), 90), cmap='gray')
# plt.colorbar(ax=ax[1])
plt.show()
