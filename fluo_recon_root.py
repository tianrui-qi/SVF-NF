# %% # import
""" import """

import torch
import numpy as np
import scipy.io

import os
import tqdm
import warnings
import tifffile 
import dataclasses
import matplotlib.pyplot as plt

from network import FullModel
from util import getPSFsim, plotz, plot_deconvolution
import src

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("medium")
# disable MPS performance warnings
warnings.filterwarnings(
    "ignore",
    message=".*MPS: The constant padding of more than 3 dimensions.*",
)

# device
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")


# %% # setup
""" setup """

# config
config = src.config.ConfigRoot()
# load
I = src.data.getI(config.data_load_path, config.N)  # (C=4,  1, H=N, W=N)
PSFexp  = src.data.getPSFexp(config.psf_load_path)  # (C=4, 41, H=K, W=K)


# %% decon with experimental PSF

# deconvolution
model = src.model.DeconPoisson(I.to(device), PSFexp.to(device))
for _ in tqdm.tqdm(range(config.epoch_decon)): model()
Oexp = model.O.detach().cpu()
del model   # free memory
# save
os.makedirs(config.data_save_fold, exist_ok=True)
tifffile.imwrite(
    os.path.join(config.data_save_fold, "Oexp.tif"),
    Oexp.detach().cpu().numpy().astype(np.float32)[:, :, None, ...],
    imagej=True,
    metadata={"axes": "TZCYX"}  # tell ImageJ how to interpret dimensions
)


# %% Recompute analytic PSF for retrieved phase

PSF, PSFR, pupil_ampli_s, pupil_ampli_p, defocus = getPSFsim(
    **dataclasses.asdict(config)
)

# %%
dzs = 1e-3 * torch.arange(
    config.z_min, config.z_max + config.z_sep, config.z_sep
).to(torch.float32)
num_z = len(dzs)
defocus_temp = defocus
defocus = (
    torch.tensor(defocus)
    .to(torch.float32)[None, None]
    .repeat(1, num_z, 1, 1)
)
# multiply defocus phase term (1, num_z, im_size, imsize) by dzs (num_z, )
defocus = defocus * dzs[..., None, None]
defocus = defocus.to(device)

PSF = PSF.permute(2, 3, 0, 1)
PSFR = PSFR.permute(2, 3, 0, 1) # this is PSF rotated by 180 degrees
num_z = PSF.shape[1]


 # %% Deconvolution with retrieved phase PSF (second stage)
abe = scipy.io.loadmat(
    config.psf_load_path
)['phase_init']
pupil_ampli_s_temp = np.moveaxis(pupil_ampli_s, -1, 0)
pupil_ampli_p_temp = np.moveaxis(pupil_ampli_p, -1, 0)
# s-polarization
pupil_ampli_s = np.moveaxis(pupil_ampli_s, -1, 0)
pupil_ampli_s = (
    torch.tensor(pupil_ampli_s)
    .to(torch.float32)[:, None]
    .repeat(1, num_z, 1, 1)
)
pupil_ampli_s = pupil_ampli_s.to(device)
# p-polarization
pupil_ampli_p = np.moveaxis(pupil_ampli_p, -1, 0)
pupil_ampli_p = (
    torch.tensor(pupil_ampli_p)
    .to(torch.float32)[:, None]
    .repeat(1, num_z, 1, 1)
)
pupil_ampli_p = pupil_ampli_p.to(device)

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

#  to tensor
abe = torch.tensor(abe).to(torch.float32).to(device) 
PSF = abe_to_psf(
    abe, config.C, pupil_ampli_s, pupil_ampli_p, defocus
)
PSF = PSF / PSF.sum() * config.C * len(dzs)

model = src.model.DeconPoisson(I.to(device), PSF.to(device))
for _ in tqdm.tqdm(range(config.epoch_decon)): model()
Oret = model.O.detach().cpu()

# show the deconvolved results
plot_deconvolution(
    num_z, Oret, config.data_save_fold, tag='Retrieved Phase Deconvolution'
)
g_retri = Oret.clone()


# %% Neural representations: merge experimental and retrieved stacks
#######################################################
 # Neural representations
dzs_ret = torch.arange(
    config.z_min, config.z_max + config.z_sep, config.z_sep
).to(torch.float32)
dzs_exp = torch.arange(-2.0, 2.0 + 0.1, 0.1).to(torch.float32)

idx_eInr = [
    torch.argmin(torch.abs(dzs_ret - dz)).item() 
    for dz in dzs_exp if torch.min(torch.abs(dzs_ret - dz)) < 0.001
]
idx_eIne = [
    torch.argmin(torch.abs(dzs_exp - dz)).item() 
    for dz in dzs_ret if torch.min(torch.abs(dzs_exp - dz)) < 0.001
]

g_retri[0,idx_eInr] = Oexp[0,idx_eIne]
idx_eNotInr = [i for i in range(len(dzs_exp)) if i not in idx_eIne]

dzs = 1e-3 * torch.cat((dzs_exp[idx_eNotInr], dzs_ret))
g = torch.cat((Oexp[0,idx_eNotInr], g_retri[0]), dim=0).unsqueeze(0)
dzs, idx = torch.sort(dzs)
g = g[:,idx]          

if config.if_log:
    g = torch.log(g + 1)
    log_gmax, log_gmin = g.max(), g.min()
    g = (g - log_gmin) / (log_gmax - log_gmin)

num_z = len(dzs)
dzs = dzs.to(device)

# %% Build neural field model (3D representation)
model = FullModel(
    w = g.shape[-1], 
    h = g.shape[-1], 
    num_feats = config.num_feats, 
    x_mode = g.shape[-1], 
    y_mode = g.shape[-2],
    z_min = dzs[0], 
    z_max = dzs[-1], 
    z_dim = config.z_dim,
    ds_factor = 2, 
    use_layernorm = config.use_layernorm
).to(device)

model_fn = model
model_fn = torch.compile(model, backend="inductor")

# %% Prepare pupil amplitudes and phase for neural optimization
# Set Pupil amplitude
pupil_ampli_s = (
    torch.tensor(pupil_ampli_s_temp)
    .to(torch.float32)[:, None]
    .repeat(1, num_z, 1, 1)
    .to(device)
)
pupil_ampli_p = (
    torch.tensor(pupil_ampli_p_temp)
    .to(torch.float32)[:, None]
    .repeat(1, num_z, 1, 1)
    .to(device)
)

abe = scipy.io.loadmat(
    config.psf_load_path
)['phase_init']
abe = torch.tensor(abe).to(torch.float32).to(device) 

# %% Optimization setup (optimizer, scheduler, loss)
num_epochs = config.learn_psf_epochs + config.init_epochs

optimizer = torch.optim.AdamW(model_fn.parameters(), lr=config.lr_psf)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=config.lr_psf // 6
)
loss_fn = torch.nn.SmoothL1Loss()

# %% Train neural field model (two-stage optimization)
tbar = tqdm.tqdm(range(num_epochs), desc='Optimization')
for epoch in tbar:

    if epoch < config.init_epochs:
        
        optimizer.zero_grad()
        g_est = model_fn(dzs) # sample the model with predefined dzs
        
        im_loss = loss_fn(g_est, g) 
        loss = im_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        tbar.set_postfix(Loss = f'{im_loss.item():.3f}')

        if config.show_inter_imgs and \
        (epoch + 1) % config.display_freq == 0:
            plt.figure(figsize=(10, 10), dpi=300)
            for i in range(num_z):
                plt.subplot(
                    int(np.sqrt(num_z) + 1), 
                    int(np.sqrt(num_z) + 1), 
                    i + 1
                )
                plt.imshow(np.clip(
                    g_est[0, i].detach().cpu().numpy(), 0, 1e3
                ), cmap='gray')
                plt.axis('image')
                plt.axis('off')
                plt.title(f"z{i}")
            plt.suptitle('Deconvolution', fontsize=16, y=0.17)
            plt.savefig(
                f'{config.data_save_fold}/Deconvolved_results_{epoch + 1}.png', dpi=300
            )
            plt.close()

    else:
        if epoch == config.init_epochs:
            PSF = scipy.io.loadmat(
                config.psf_load_path
            )['PSF']
            PSFR = scipy.io.loadmat(
                config.psf_load_path
            )['PSFR']
            PSF = torch.tensor(PSF).to(torch.float32)
            PSFR = torch.tensor(PSFR).to(torch.float32)
            PSF = PSF.permute(2, 3, 0, 1)
            PSFR = PSFR.permute(2, 3, 0, 1) 

            exp_idx = [
                torch.argmin(torch.abs(dzs - dz)).item() 
                for dz in dzs 
                if torch.min(torch.abs(dzs_exp*1e-3 - dz)) < 1e-5
            ]
            ret_idx = [i for i in range(len(dzs)) if i not in exp_idx]
            
        optimizer.zero_grad()
        dzs_sample = (
            1e-3 * dzs[exp_idx] + 
            0.5e-4 * torch.rand(len(dzs[exp_idx])) 
        ).to(torch.float32)
        dzs_ext = (
            1e-3 * dzs[ret_idx] + 
            0.5e-4 * torch.rand(len(dzs[ret_idx])) 
        ).to(torch.float32)

        g_est = model_fn(dzs)
        g_ret_sample = g_est[:, ret_idx]
        g_exp_sample = g_est[:, exp_idx]

        z_data = model.model_3D.img_real.z_data
        sparsity_loss = (
            config.l1_z * torch.mean(z_data.abs()) + 
            config.l1_g * torch.mean(g_est.abs())
        )

        g_est = g_exp_sample
        if config.if_log:
            g_est = g_est * (log_gmax - log_gmin) + log_gmin
            g_est = torch.exp(g_est) - 1
        F_g_est = torch.fft.fftn(g_est, dim=(-2, -1))

        # experimental PSF
        psf_size, pad_size1 = PSF.shape[-1], g.shape[-1]  - PSF.shape[-1]
        psf_padded = torch.nn.functional.pad(
            PSF, (0, pad_size1,  0, pad_size1)
        )
        psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1))

        FI_est = psf_fft * F_g_est.repeat(config.C, 1, 1, 1)
        I_est_sample = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs()

        # retrieved PSF
        g_est = g_ret_sample
        
        if config.if_log:
            g_est = g_est * (log_gmax - log_gmin) + log_gmin
            g_est = torch.exp(g_est) - 1
        F_g_est = torch.fft.fftn(g_est, dim=(-2, -1))


        # Set defocus phase term
        defocus_ext = (
            torch.tensor(defocus_temp)
            .to(torch.float32)[None, None]
            .repeat(1, len(dzs_ext), 1, 1)
        )
        # multiply defocus phase term 
        # (1, num_z, im_size, imsize) by dzs (num_z, )
        defocus_ext = defocus_ext * dzs_ext[..., None, None]
        defocus_ext = defocus_ext.to(device)
        
        pupil_ampli_s = pupil_ampli_s[:, :len(dzs_ext)] 
        pupil_ampli_p = pupil_ampli_p[:, :len(dzs_ext)] 

        psf = abe_to_psf(
            abe, config.C, pupil_ampli_s, pupil_ampli_p, defocus_ext
        )
        psf = psf / psf.sum() * config.C * psf.size(1)
        psf_size, pad_size1 = psf.shape[-1], g.shape[-1]  - psf.shape[-1]
        psf_padded = torch.nn.functional.pad(
            psf, (0, pad_size1,  0, pad_size1)
        )
        psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1))

        FI_est = psf_fft * F_g_est.repeat(config.C, 1, 1, 1)
        I_est_ext = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs()

        # concatenate the two results
        I_est = torch.cat((I_est_sample, I_est_ext), dim=1)
        # sum pooling
        I_est = torch.sum(I_est, dim=1, keepdim=True)
        I_est = torch.roll(
            I_est, shifts=(-psf_size // 2 + 1, -psf_size // 2 + 1), 
            dims=(2,3)
        )

        im_loss = loss_fn(I_est, I) / I.mean()
        loss = im_loss + sparsity_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        tbar.set_postfix(Loss = f'{im_loss.item():.3f}')


        if config.show_inter_imgs and \
        (epoch + 1) % config.display_freq == 0:
            ext = 1
            sample_slice = int(num_z * ext)
            dz_sample = 1e-3 * torch.arange(
                config.z_min * ext, 
                config.z_max * ext, 
                ext * (config.z_max - config.z_min) / sample_slice
            ).to(torch.float32)
            g_sample = model_fn(dz_sample)
            if config.if_log:
                g_sample = g_sample * (log_gmax - log_gmin) + log_gmin
                g_sample = torch.exp(g_sample) - 1
            
            plt.figure(figsize=(10, 10), dpi=300)
            for i in range(num_z):
                plt.subplot(
                    int(np.sqrt(num_z) + 1), 
                    int(np.sqrt(num_z) + 1), 
                    i + 1
                )
                plt.imshow(np.clip(
                    g_sample[0, i].detach().cpu().numpy(), 0, 1e3
                ), cmap='gray')
                plt.axis('image')
                plt.axis('off')
                plt.title(f"z{i}")
            plt.suptitle('Deconvolution', fontsize=16, y=0.17)
            plt.savefig(
                f'{config.data_save_fold}/Deconvolved_results_{epoch + 1}.png', dpi=300
            )
            plt.close()

            # Show estimated images
            plt.figure(figsize=(10, 10),dpi=300)
            for i in range(config.C):
                plt.subplot(2, 2, i+1)
                plt.imshow(
                    torch.relu(I_est[i, 0]).detach().cpu().numpy(), 
                    cmap='gray'
                )
                plt.axis('image')
                plt.axis('off')
                plt.title(f"p{i}")
            plt.suptitle('Estimated measurements', fontsize=16, y=0.93)
            plt.savefig(
                f'{config.data_save_fold}/Estimated_measurements_{epoch + 1}.png', 
                dpi=300
            )
            plt.close()

# %% Post-process neural field output (undo log transform)
if config.if_log:
    g = g * (log_gmax - log_gmin) + log_gmin
    g = torch.exp(g) - 1

# %% Sample trained neural field densely along z for visualization
ext = 1
sample_slice = int(num_z * ext)
dz_sample = 1e-3 * torch.arange(
    config.z_min * ext, config.z_max * ext, 
    ext * (config.z_max - config.z_min) / sample_slice
).to(torch.float32)
g_sample = model_fn(dz_sample)
if config.if_log:
    g_sample = g_sample * (log_gmax - log_gmin) + log_gmin
    g_sample = torch.exp(g_sample) - 1


# %% Visualize and export reconstructed volume stacks
gPlot_filt_1 = plotz(
    g.detach().cpu().numpy(), config.data_save_fold, 'Stack Deconvolution'
)
gPlot_filt_2 = plotz(
    g_sample.detach().cpu().numpy(), config.data_save_fold, 'Stack INR'
)

# plt.figure(dpi = 400)
# plt.imshow(gPlot_filt_1)
# plt.axis('off')
# plt.title('Stack Deconvolution')
# plt.savefig(f'{config.data_save_fold}/Stack Deconvolution.png', dpi=400)
# plt.close()

# plt.figure(dpi = 400)
# plt.imshow(gPlot_filt_2)
# plt.axis('off')
# plt.title('Stack INR')
# plt.savefig(f'{config.data_save_fold}/Stack INR.png', dpi=400)
# plt.close()
