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

import src

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("medium")
# disable MPS performance warnings
warnings.filterwarnings("ignore", message=".*MPS:*")

# device
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")


# %% # setup
""" setup """

# config
config = src.config.Config()
# measurement
I = src.data.getI(config.data_load_path, config.N)  # (C=4,  1, H=N, W=N)


# %% # deconvolution with experimental psf
""" deconvolution with experimental psf """

# psf
PSFexp  = src.data.getPSFexp(config.psf_load_path)  # (C=4, 41, H=K, W=K)
# deconvolution
model = src.model.DeconRL(I.to(device), PSFexp.to(device))
for _ in tqdm.tqdm(range(config.epoch_decon), desc="deconvolution"): model()
Oexp = model.O.detach().cpu()
# save
os.makedirs(config.data_save_fold, exist_ok=True)
filename = config.data_load_path.split('/')[-1].split('.')[0]
tifffile.imwrite(
    os.path.join(config.data_save_fold, f"{filename}_Oexp.tif"),
    Oexp.numpy().astype(np.float32)[:, :, None, ...],
    imagej=True, metadata={"axes": "TZCYX"}
)
# clean
del model
if device.type == 'cuda': torch.cuda.empty_cache()
if device.type == 'mps': torch.mps.empty_cache()

# %% # deconvolution with retrieved psf
""" deconvolution with retrieved psf """

# psf
PSFsim, pupil_ampli_s, pupil_ampli_p, defocus = src.data.getPSFsim(
    **dataclasses.asdict(config), z_min=config.z_ret[0], z_max=config.z_ret[1]
)
PSFret = src.data.getPSFret(
    PSFsim, pupil_ampli_s, pupil_ampli_p, defocus,
    **dataclasses.asdict(config), z_min=config.z_ret[0], z_max=config.z_ret[1]
)
# deconvolution
model = src.model.DeconRL(I.to(device), PSFret.to(device))
for _ in tqdm.tqdm(range(config.epoch_decon), desc="deconvolution"): model()
Oret = model.O.detach().cpu()
# save
os.makedirs(config.data_save_fold, exist_ok=True)
filename = config.data_load_path.split('/')[-1].split('.')[0]
tifffile.imwrite(
    os.path.join(config.data_save_fold, f"{filename}_Oret.tif"),
    Oret.numpy().astype(np.float32)[:, :, None, ...],
    imagej=True, metadata={"axes": "TZCYX"}
)
# clean
del model
if device.type == 'cuda': torch.cuda.empty_cache()
if device.type == 'mps': torch.mps.empty_cache()


# %% # neural field
""" neural field """

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

dzs_ret = torch.arange(
    config.z_ret[0], config.z_ret[1] + config.z_sep, config.z_sep
).to(torch.float32)
dzs_exp = torch.arange(
    config.z_exp[0], config.z_exp[1] + config.z_sep, config.z_sep
).to(torch.float32)

idx_eInr = [
    torch.argmin(torch.abs(dzs_ret - dz)).item() 
    for dz in dzs_exp if torch.min(torch.abs(dzs_ret - dz)) < 0.001
]
idx_eIne = [
    torch.argmin(torch.abs(dzs_exp - dz)).item() 
    for dz in dzs_ret if torch.min(torch.abs(dzs_exp - dz)) < 0.001
]

g_retri = Oret.clone()
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

# model = src.model.Render3D(
#     H = g.shape[-2]//2,
#     W = g.shape[-1]//2,
#     downsample=1,
#     Dd=config.Dd,
#     z_min=float(dzs[0]),
#     z_max=float(dzs[-1]),
#     Q=config.Q,
#     hidden_dim=32,
#     num_layers=2,
#     layernorm=config.use_layernorm,
# ).to(device)
model = src.model.FullModel(
    w = g.shape[-1], 
    h = g.shape[-1], 
    Q = config.Q, 
    x_mode = g.shape[-1], 
    y_mode = g.shape[-2],
    z_min = dzs[0], 
    z_max = dzs[-1], 
    z_dim = config.Dd,
    ds_factor = 2, 
    use_layernorm = config.use_layernorm
).to(device)

model_fn = model

pupil_ampli_s = pupil_ampli_s.repeat(1, num_z, 1, 1)
pupil_ampli_p = pupil_ampli_p.repeat(1, num_z, 1, 1)

abe = torch.tensor(
    scipy.io.loadmat(config.psf_load_path)['phase_init']
).to(torch.float32).to(device)

exp_idx = [
    torch.argmin(torch.abs(dzs - dz)).item() 
    for dz in dzs if torch.min(torch.abs(dzs_exp*1e-3 - dz)) < 1e-5
]
ret_idx = [i for i in range(len(dzs)) if i not in exp_idx]

num_epochs = config.epoch_stage2 + config.epoch_stage1

optimizer = torch.optim.AdamW(model_fn.parameters(), lr=config.lr_psf)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=config.lr_psf // 6
)


# %% # first stage training
""" first stage training """

tbar = tqdm.tqdm(range(config.epoch_stage1))
for epoch in tbar:
    optimizer.zero_grad()
    g_est = model_fn(dzs.to(device)) # sample the model with predefined dzs
    # g_est = torch.nn.functional.interpolate(
    #     g_est.unsqueeze(0), size=g.shape[-2:], mode='bilinear'
    # )
    
    im_loss = torch.nn.functional.smooth_l1_loss(g_est, g.to(device)) 
    im_loss.backward()
    optimizer.step()
    scheduler.step()
    tbar.set_postfix(Loss = f'{im_loss.item():.3f}')

    if config.display_freq > 0 and (epoch + 1) % config.display_freq == 0:
        tifffile.imwrite(
            os.path.join(
                config.data_save_fold, f"{filename}_Onf_1_{epoch + 1}.tif"
            ),
            (
                torch.exp(g_est * (log_gmax - log_gmin) + log_gmin) - 1
            ).detach().cpu().numpy().astype(np.float32)[:, :, None, ...],
            imagej=True, metadata={"axes": "TZCYX"}
        )

# %% # second stage training
""" second stage training """

tbar = tqdm.tqdm(range(config.epoch_stage2))
for epoch in tbar:        
    optimizer.zero_grad()
    dzs_sample = (
        1e-3 * dzs[exp_idx].to(device) + 
        0.5e-4 * torch.rand(len(dzs[exp_idx]), device=device) 
    ).to(torch.float32)
    dzs_ext = (
        1e-3 * dzs[ret_idx].to(device) + 
        0.5e-4 * torch.rand(len(dzs[ret_idx]), device=device) 
    ).to(torch.float32)

    g_est = model_fn(dzs.to(device))
    # g_est = torch.nn.functional.interpolate(
    #     g_est.unsqueeze(0), size=g.shape[-2:], mode='bilinear'
    # )

    if config.display_freq > 0 and (epoch + 1) % config.display_freq == 0:
        tifffile.imwrite(
            os.path.join(
                config.data_save_fold, f"{filename}_Onf_2_{epoch + 1}.tif"
            ),
            (
                torch.exp(g_est * (log_gmax - log_gmin) + log_gmin) - 1
            ).detach().cpu().numpy().astype(np.float32)[:, :, None, ...],
            imagej=True, metadata={"axes": "TZCYX"}
        )

    g_ret_sample = g_est[:, ret_idx]
    g_exp_sample = g_est[:, exp_idx]

    # z_data = model.U.U
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
    psf_size, pad_size1 = PSFexp.shape[-1], g.shape[-1]  - PSFexp.shape[-1]
    psf_padded = torch.nn.functional.pad(
        PSFexp, (0, pad_size1,  0, pad_size1)
    )
    psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1)).to(device)

    FI_est = psf_fft * F_g_est.repeat(config.C, 1, 1, 1)
    I_est_sample = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs()

    # retrieved PSF
    g_est = g_ret_sample
    
    if config.if_log:
        g_est = g_est * (log_gmax - log_gmin) + log_gmin
        g_est = torch.exp(g_est) - 1
    F_g_est = torch.fft.fftn(g_est, dim=(-2, -1))

    # Set defocus phase term
    defocus_ext = defocus.repeat(1, len(dzs_ext), 1, 1).to(device)
    # multiply defocus phase term 
    # (1, num_z, im_size, imsize) by dzs (num_z, )
    defocus_ext = defocus_ext * dzs_ext[..., None, None]
    
    pupil_ampli_s = pupil_ampli_s[:, :len(dzs_ext)].to(device)
    pupil_ampli_p = pupil_ampli_p[:, :len(dzs_ext)].to(device)

    psf = abe_to_psf(
        abe, config.C, pupil_ampli_s, pupil_ampli_p, defocus_ext
    )
    psf = psf / psf.sum() * config.C * psf.size(1)
    psf_size, pad_size1 = psf.shape[-1], g.shape[-1]  - psf.shape[-1]
    psf_padded = torch.nn.functional.pad(
        psf, (0, pad_size1,  0, pad_size1)
    ).to(device)
    psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1)).to(device)

    FI_est = psf_fft * F_g_est.repeat(config.C, 1, 1, 1)
    I_est_ext = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs().to(device)

    # concatenate the two results
    I_est = torch.cat((I_est_sample, I_est_ext), dim=1)
    # sum pooling
    I_est = torch.sum(I_est, dim=1, keepdim=True)
    I_est = torch.roll(
        I_est, shifts=(-psf_size // 2 + 1, -psf_size // 2 + 1), 
        dims=(2,3)
    )

    im_loss = torch.nn.functional.smooth_l1_loss(
        I_est, I.to(device)
    ) / I.to(device).mean()
    loss = im_loss + sparsity_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    tbar.set_postfix(Loss = f'{im_loss.item():.3f}')


# %% # save
""" save """

ext = 1
sample_slice = int(num_z * ext)
dz_sample = 1e-3 * torch.arange(
    config.z_ret[0] * ext, config.z_ret[1] * ext, 
    ext * (config.z_ret[1] - config.z_ret[0]) / sample_slice
).to(torch.float32)
g_sample = model_fn(dz_sample.to(device))
if config.if_log:
    g_sample = torch.exp(g_sample * (log_gmax - log_gmin) + log_gmin) - 1

tifffile.imwrite(
    os.path.join(
        config.data_save_fold, f"{filename}_Onf.tif"
    ),
    g_sample.detach().cpu().numpy().astype(np.float32)[:, :, None, ...],
    imagej=True, metadata={"axes": "TZCYX"}
)

# %%
