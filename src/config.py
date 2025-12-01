import dataclasses


@dataclasses.dataclass(slots=True)
class Config():
    # path
    psf_load_path: str = "data/ExpPSF_605.mat"
    data_load_path: str = ""
    data_save_fold: str = ""

    # dimension
    C: int = 4      # number of polarizations
    N: int = 1024   # image H and W after channel seperation
    K: int = 256    # PSF H and W
    z_max: float =  4.0     # maximum z-value in mm
    z_min: float = -4.0     # minimum z-value in mm
    z_sep: float = 0.1      # z separation in mm
    z_dim: int = 8

    # system 
    f_obj: float = 80e-3        # objective lens focal length
    f_tube: float = 150e-3      # tube lens focal length
    px_size: float = 6.9e-6     # camera pixel size
    wavelength: float = 605e-9  # emission wavelength
    NA: float = 0.0563          # numerical aperture
    block_line: float = 0.6e-3  # 3D printed pupil block center line width
    pol_dir: tuple[int, ...] = (1, 2, 3, 0)     # polarizer direction

    # run
    epoch_decon: int = 100  # num of iterations for RL deconvolution

    # HACK

    # show and save intermediate images
    show_inter_imgs: bool = True
    # display / save intermediate image frequency, every n epochs
    display_freq: int = 100

    # Learn PSF options
    if_log: bool = False            # logarithm of raw image
    if_lr_psf: bool = True          # if learn PSF
    learn_psf_epochs: int = 100     # number of epochs for learn psf
    init_epochs: int = 100          # number of epochs for initialization
    lr_psf: float = 8e-3            # learning rate for learn psf
    use_layernorm: bool = False     # use layernorm in PSF model

    # Learn PSF parameters
    num_coeff: int = 100            # number of Zernike coefficients
    num_feats: int = 32             # number of features in the network
    l1_g: float = 0.0               # L1 sparsity weight for g_est
    l1_z: float = 0.0               # L1 sparsity weight for z_data


class ConfigRoot(Config):
    def __init__(self) -> None:
        super().__init__()
        self.data_load_path = "data/Roots_xyScan_128.tif"
        self.data_save_fold = "data/root/"
        self.show_inter_imgs = False
        self.if_log = True
        self.lr_psf = 5e-3


class ConfigLymphn(Config):
    def __init__(self) -> None:
        super().__init__()
        self.data_load_path = "data/lymphn.tif"
        self.data_save_fold = "data/lymphn/"
        self.if_log = True
        self.lr_psf = 5e-3
