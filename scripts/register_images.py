import numpy as np 
import cupy as cp
import os
import sys
from natsort import natsorted
import skimage.registration as reg
from skimage.transform import warp
from cucim.skimage.transform import warp
import cucim.skimage.registration as reg
from scipy.stats import pearsonr
from utils import load_utils
from utils import image_preprocessing_parallel_dev as img_pre

registration_channel = '89Y_Collagen_type_I'


input_dir=sys.argv[1]
output_dir=input_dir[:-1]+'_registered/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def plot_diff(im1, im2):
    im1[im1>1] = 1
    im2[im2>1] = 1
    combined_image = (np.dstack((cp.asnumpy(im1), cp.asnumpy(im2), cp.asnumpy(im2)))*255).astype(np.uint8)
    
    return combined_image

def plot_diffs(ims, ncol=5):
    nrow = int(np.ceil((len(ims)-1)/ncol))
    _, ax = plt.subplots(nrow, ncol, figsize=(20,8))
    ax = ax.ravel()
    [a.axis('off') for a in ax]
    for i, (im1, im2) in enumerate(zip(ims[:-1], ims[1:])):
        ax[i].imshow(plot_diff(im1, im2))
    plt.tight_layout()
    plt.show()


def pad_image(im, target_h, target_w):
    
    h, w = im.shape

    y_dif = target_h-h
    y_b4 = y_dif//2
    y_after = y_dif-y_b4

    x_dif = target_w-w
    x_b4 = x_dif//2
    x_after = x_dif-x_b4

    return cp.pad(im, ((y_b4, y_after), (x_b4, x_after)))

def calc_mean_corrs(ims):
    return np.mean([pearsonr(cp.asnumpy(im1).ravel(), cp.asnumpy(im2).ravel()).statistic for  im1,im2 in zip(ims[:-1], ims[1:])])


im_dirs = natsorted(os.listdir(input_dir))
ims = [cp.array(load_utils.load_channel(os.path.join(input_dir, td), registration_channel)) for td in im_dirs]
max_h = np.max([im.shape[0] for im in ims])
max_w = np.max([im.shape[1] for im in ims])
ims = [pad_image(im, max_h, max_w) for im in ims]
ims = [img_pre.normalize_channels_mat_cp(img_pre.normalize_channels_mat_cp(im, method='arcsinh', cofactor=None), method='quantile') for im in ims]

middle_idx = len(ims)//2

# Register images
coords = {}
middle_name = im_dirs[middle_idx]
coords[middle_name] = {}
coords[middle_name]['u'] = 0
coords[middle_name]['v'] = 0
reg_func = reg.optical_flow_tvl1

j = 0
for i in range(middle_idx, 0, -1):
    h, w = ims[i-1].shape
    row_coords, col_coords = cp.meshgrid(cp.arange(h), cp.arange(w), indexing='ij')
    v, u = reg_func(ims[i], ims[i-1],attachment=15, tightness=0.5, num_warp=5, num_iter=10, tol=0.0001)
    ims[i-1] = warp(ims[i-1], cp.array([row_coords + v, col_coords + u]), mode='edge')
    coords[im_dirs[i-1]] = {}
    coords[im_dirs[i-1]]['v'] = v
    coords[im_dirs[i-1]]['u'] = u

    j+=1
    print(f'{j}/{len(ims)-1} registrations complete')

for i in range(middle_idx, len(ims)-1):
    h, w = ims[i+1].shape
    row_coords, col_coords = cp.meshgrid(cp.arange(h), cp.arange(w), indexing='ij')
    v, u = reg_func(ims[i], ims[i+1])
    ims[i+1] = warp(ims[i+1], cp.array([row_coords + v, col_coords + u]), mode='edge')
    coords[im_dirs[i+1]] = {}
    coords[im_dirs[i+1]]['v'] = v
    coords[im_dirs[i+1]]['u'] = u

    j+=1
    print(f'{j}/{len(ims)-1} registrations complete')

# Call registration function
print('Registration completed')

# Save registered images

from utils import save_utils
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
row_coords, col_coords = cp.meshgrid(cp.arange(max_h), cp.arange(max_w), indexing='ij')
for im, td in zip(ims, im_dirs):
    if not os.path.exists(os.path.join(output_dir, td)):
        os.mkdir(os.path.join(output_dir, td))
    for chan in os.listdir(os.path.join(input_dir, td)):
        tif = cp.array(load_utils.load_tiff(os.path.join(input_dir, td, chan)))
        tif = pad_image(tif, max_h, max_w)
        tif = warp(tif, cp.array([row_coords+coords[td]['v'], col_coords+coords[td]['u']]), mode='edge', preserve_range=True)
        tif = cp.asnumpy(tif)
        tif = (tif+0.5).astype(np.uint16) #round and convert back to int
        save_utils.save_tiff(os.path.join(output_dir, td, chan), tif)
print(f'Channels saved')

