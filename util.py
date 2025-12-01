import numpy as np
import scipy.ndimage

import os
import colorsys
import matplotlib.pyplot as plt


def rgb2hls(rgb):
    return np.array([colorsys.rgb_to_hls(*color[:3]) for color in rgb])


def hls2rgb(hls):
    return np.array([colorsys.hls_to_rgb(*color) for color in hls])


def cMapHeight_hsv_v2(N, s):
    hsv_map = plt.cm.hsv(np.linspace(0, 1, 900))[:600, :3]
    hls_map = rgb2hls(hsv_map)
    hls_map[:, 1] = s # Modify the lightness in HLS
    rgb_map = hls2rgb(hls_map)
    idx = np.round(np.linspace(0, 599, N)).astype(int)
    out_cmap = rgb_map[idx]
    return out_cmap


def plotz(g, out_dir, title="Stack Ground Truth"):
    num_channels = g.shape[1]
    middle_portion_index = int(num_channels * 0.625)
    end_portion_index = (num_channels - middle_portion_index) // 2

    # Create a colormap that spans 62.5% of the range
    cMap = cMapHeight_hsv_v2(middle_portion_index, .6)

    # Initialize the full colormap array
    full_cMap = np.zeros((num_channels, 3))

    # Set the start and end portions to the same color as the edges of the middle portion
    full_cMap[:end_portion_index] = cMap[0]
    full_cMap[-end_portion_index-1:] = cMap[-1]

    # Fill the middle portion with the actual colormap
    list_cMap = np.arange(end_portion_index,num_channels-end_portion_index-1,1)
    if len(cMap) == len(list_cMap):
        full_cMap[end_portion_index:num_channels-end_portion_index-1] = cMap
    else:
        full_cMap[end_portion_index:num_channels-end_portion_index] = cMap


    gPlot = np.zeros((g.shape[2], g.shape[3], 3))
    for i in range(num_channels):
        colorized_channel = np.stack([
            full_cMap[i, 0] * g[0, i, :, :],
            full_cMap[i, 1] * g[0, i, :, :],
            full_cMap[i, 2] * g[0, i, :, :]
        ], axis=-1)
        gPlot += colorized_channel

    normfac = 6e3 / (2**16-1) * 1.5e5
    if title == "Stack Ground Truth":
        gPlot_filt = scipy.ndimage.gaussian_filter(gPlot, sigma=5, axes=(0,1))
        gPlot_filt = gPlot_filt /normfac
    else:
        gPlot_filt = gPlot /normfac

    plt.figure(dpi=400)
    plt.imshow(gPlot_filt)
    plt.axis('off')
    plt.title(title)
    plt.savefig(os.path.join(out_dir, title + '.png'), dpi=400)

    return gPlot_filt


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi,rho)
