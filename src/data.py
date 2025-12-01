import torch
import numpy as np
import scipy.ndimage

import skimage.io
import scipy.io


def getI(data_load_path: str, N: int) -> torch.Tensor:
    # load measurement
    # note that this image's dimension may not exactly be (2N, 2N)
    I = torch.tensor(       # (H=2048, W=2448)
        skimage.io.imread(data_load_path)
    ).float()
    # channel separation
    I = torch.stack([       # (C=4, H//2, W//2)
        I[ ::2,  ::2],      # [0, 0]: 0°
        I[ ::2, 1::2],      # [0, 1]: 45°
        I[1::2, 1::2],      # [1, 1]: 90°
        I[1::2,  ::2],      # [1, 0]: 135°
    ], dim=0)
    # crop
    I = I[:, :N, :N]        # (C=4, H=N, W=N)
    I = I.unsqueeze(1)      # (C=4, D=1, H=N, W=N)
    # denoise
    # apply 5×5 median filter over spatial dims (H, W) to
    # suppress local spike noise to prevent RL deconvolution and neural field 
    # fitting from amplifying it, while preserving edges and fine structures
    I = torch.tensor(
        scipy.ndimage.median_filter(I.numpy(), size=(1, 1, 5, 5))
    ).float()
    # return
    return I    # (C=4, D=1, H=N, W=N)


def getPSFexp(psf_load_path: str) -> torch.Tensor:
    # NOTE: assume PSF store in 'PSF' of .mat file in order (H, W, C, D)
    # psf
    PSF = torch.tensor(     # (C, D, H=K, W=K)
        scipy.io.loadmat(psf_load_path)['PSF']
    ).float().permute(2, 3, 0, 1)
    # global energy normalization
    PSF = PSF / PSF.sum() * PSF.size(0) * PSF.size(1)
    # return
    return PSF  # (C, D, H=K, W=K)


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
    PSF = torch.tensor(PSF).to(torch.float32).permute(2, 3, 0, 1)

    # s-polarization
    pupil_amplitude_s_pol = np.moveaxis(pupil_amplitude_s_pol, -1, 0)
    pupil_amplitude_s_pol = (
        torch.tensor(pupil_amplitude_s_pol)
        .to(torch.float32)[:, None]
    )
    # p-polarization
    pupil_amplitude_p_pol = np.moveaxis(pupil_amplitude_p_pol, -1, 0)
    pupil_amplitude_p_pol = (
        torch.tensor(pupil_amplitude_p_pol)
        .to(torch.float32)[:, None]
    )
    # defocus
    p_defocus = torch.tensor(p_defocus).to(torch.float32)[None, None]

    return (
        PSF, 
        pupil_amplitude_s_pol, pupil_amplitude_p_pol, 
        p_defocus
    )


def getPSFret(
    PSF: torch.Tensor, 
    pupil_ampli_s: torch.Tensor, pupil_ampli_p: torch.Tensor, 
    defocus: torch.Tensor,
    psf_load_path: str, C: int,
    z_min: float, z_max: float, z_sep: float, 
    **kwargs
) -> torch.Tensor:
    dzs = 1e-3 * torch.arange(z_min, z_max + z_sep, z_sep)
    defocus = defocus.repeat(1, len(dzs), 1, 1)
    defocus = defocus * dzs[..., None, None]
    # s-polarization
    pupil_ampli_s = pupil_ampli_s.repeat(1, PSF.shape[1], 1, 1)
    # p-polarization
    pupil_ampli_p = pupil_ampli_p.repeat(1, PSF.shape[1], 1, 1)
    # aberration
    abe = scipy.io.loadmat(psf_load_path)['phase_init']
    abe = torch.tensor(abe).to(torch.float32)

    # convert aberration to PSF
    def abe_to_psf(aberration, num_pol, pupil_ampli_s, pupil_ampli_p, defocus):
        pupil_phase = (aberration + defocus).repeat(num_pol, 1, 1, 1)
        pupil_s = pupil_ampli_s * torch.exp(1j * pupil_phase) 
        pupil_p = pupil_ampli_p * torch.exp(1j * pupil_phase) 
        pupil_ifft = torch.flip(torch.fft.ifftshift(torch.fft.ifftn(
            pupil_s, dim=(-2, -1)
        ), dim=(-2, -1)), dims=[-2, -1])
        psf_s = torch.abs(pupil_ifft) ** 2
        pupil_ifft = torch.flip(torch.fft.ifftshift(torch.fft.ifftn(
            pupil_p, dim=(-2, -1)
        ), dim=(-2, -1)), dims=[-2, -1])
        psf_p = torch.abs(pupil_ifft) ** 2

        psf = psf_s + psf_p
        return psf

    PSF = abe_to_psf(
        abe, C, pupil_ampli_s, pupil_ampli_p, defocus
    )
    PSF = PSF / PSF.sum() * C * len(dzs)

    return PSF
