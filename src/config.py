import dataclasses


@dataclasses.dataclass(slots=True)
class Config():
    # path
    psf_load_path: str = "data/psf.mat"
    data_load_path: str = "data/frame/Roots_xyScan_128.tif"
    data_save_fold: str = "data/result/"

    # dimension
    C: int = 4      # number of polarizations
    N: int = 1024   # image H and W after channel seperation
    K: int = 256    # PSF H and W
    Dd: int = 8     # number of depth for neural field
    z_exp: tuple[float, float] = (-2.0, 2.0)    # depth range in mm
    z_ret: tuple[float, float] = (-4.0, 4.0)    # depth range in mm
    z_sep: float = 0.1                          # z separation in mm
    Q: int = 32     # dimension of feature for neural field
    
    # system 
    f_obj: float = 80e-3        # objective lens focal length
    f_tube: float = 150e-3      # tube lens focal length
    px_size: float = 6.9e-6     # camera pixel size
    wavelength: float = 605e-9  # emission wavelength
    NA: float = 0.0563          # numerical aperture
    block_line: float = 0.6e-3  # 3D printed pupil block center line width
    pol_dir: tuple[int, ...] = (1, 2, 3, 0)     # polarizer direction

    # run
    epoch_decon: int = 100      # num of iterations for RL deconvolution
    epoch_stage1: int = 100     # epochs for initialization
    epoch_stage2: int = 100     # epochs for learn psf
    display_freq: int = 10      # save result every n epochs

    # Learn PSF options
    if_log: bool = True             # logarithm of raw image
    if_lr_psf: bool = True          # if learn PSF
    lr_psf: float = 2e-3            # learning rate for learn psf
    use_layernorm: bool = False     # use layernorm in PSF model
    # Learn PSF parameters
    num_coeff: int = 100            # number of Zernike coefficients
    l1_g: float = 0.0               # L1 sparsity weight for g_est
    l1_z: float = 0.0               # L1 sparsity weight for z_data
