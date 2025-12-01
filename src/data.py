import torch
import skimage.io
import scipy.ndimage


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


def getPSFsim():
    pass
