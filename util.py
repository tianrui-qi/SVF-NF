import torch
import numpy as np
from scipy.ndimage import gaussian_filter

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
        gPlot_filt = gaussian_filter(gPlot, sigma=5, axes=(0,1))
        gPlot_filt = gPlot_filt /normfac
    else:
        gPlot_filt = gPlot /normfac

    plt.figure(dpi=400)
    plt.imshow(gPlot_filt)
    plt.axis('off')
    plt.title(title)
    plt.savefig(os.path.join(out_dir, title + '.png'), dpi=400)

    return gPlot_filt


def plot_deconvolution(num_z, g, out_dir,tag='Theoretical PSF Deconvolution'):
    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(num_z):
        plt.subplot(int(np.sqrt(num_z) + 1), int(np.sqrt(num_z) + 1), i + 1)
        plt.imshow(np.clip(g[0, i].detach().cpu().numpy(), 0, 1e3), cmap='gray')
        plt.axis('image')
        plt.axis('off')
        plt.title(f"z{i}")
    plt.suptitle('Deconvolution', fontsize=16, y=0.17)
    plt.savefig(f'{out_dir}/Results '+tag+'.png', dpi=300)

    return None


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi,rho)


def getPSFsim(
    f_obj: float, f_tube: float, px_size: float, wavelength: float, NA: float,
    block_line: float, pol_dir: tuple[int, ...], 
    z_min: float, z_max: float, z_sep: float, K: int, C: int,
    **kwargs
):
    M = f_tube / f_obj  # magnification
    rBFP = NA * f_obj   # pupil radius
    rBFP_px = round(    # pupil diameter in pixels
        K * NA * px_size / wavelength / M
    )

    # 3center line half width in pixels
    block_line_px = round(rBFP_px / rBFP / 2 * block_line)  

    u, v = np.meshgrid(
        np.linspace(
            -wavelength / (2 * px_size / M), 
            wavelength / (2 * px_size / M), K
        ),
        np.linspace(
            -wavelength / (2 * px_size / M),
            wavelength / (2 * px_size / M), K
        )
    )
    p, r = np.arctan2(v, u), np.hypot(u, v)

    # compute amplitude mask A
    Atmp = np.ones((K, K))
    Atmp[r >= NA] = 0
    Atmp[K // 2 - block_line_px:K // 2 + block_line_px, :] = 0
    Atmp[:, K // 2 - block_line_px:K // 2 + block_line_px] = 0

    pupil_amplitude_s_pol = np.zeros((K, K, C))
    pupil_amplitude_p_pol = np.zeros((K, K, C))
    for i in range(C):
        pupil_amplitude_s_pol[:, :, i] = Atmp * np.rot90(
            np.block([
                [
                    np.ones((K // 2, K // 2)), 
                    0.5 * np.ones((K // 2, K // 2))
                ],
                [
                    0.5 * np.ones((K // 2, K // 2)), 
                    np.zeros((K // 2, K // 2))
                ]
            ]), k=pol_dir[i]
        )
        pupil_amplitude_p_pol[:, :, i] = Atmp * np.rot90(
            np.block([
                [
                    np.zeros((K // 2, K // 2)), 
                    0.5 * np.ones((K // 2, K // 2))
                ],
                [
                    -0.5 * np.ones((K // 2, K // 2)), 
                    np.zeros((K // 2, K // 2))
                ]
            ]), k=pol_dir[i]
        )
        
    # defocus phase
    p_defocus = 2 * np.pi / wavelength * np.cos(np.arcsin(r))
    p_defocus = p_defocus - np.min(p_defocus)

    # FT to find PSF
    # for z in ... A[:,:,0] -- first channel, etc.
    dzs = 1e-3 * np.arange(z_min, z_max + z_sep, z_sep)
    # print(len(dzs))
    PSF = np.zeros((K, K, C, len(dzs)))
    
    for pol in range(C):
        count = 0
        for z in dzs:
            PSF[:,:,pol,count] = (
                np.abs(np.fft.fftshift(np.fft.fft2(
                    pupil_amplitude_s_pol[:, :, pol] * 
                    np.exp(1j * p_defocus * z)
                )))**2 + 
                np.abs(np.fft.fftshift(np.fft.fft2(
                    pupil_amplitude_p_pol[:, :, pol] * 
                    np.exp(1j * p_defocus * z)
                )))**2
            )
            count += 1

    PSF = PSF / np.sum(PSF) * C * len(dzs)

    # rotate PSF for 180 deg at each plane (axis=2) [N N num_pol 31]
    PSFR = np.rot90(PSF, k=2, axes=(0,1))

    return (
        torch.tensor(PSF.copy()).to(torch.float32), 
        torch.tensor(PSFR.copy()).to(torch.float32), 
        pupil_amplitude_s_pol, 
        pupil_amplitude_p_pol, 
        p_defocus
    )
