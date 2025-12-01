import torch


class DeconPoisson(torch.nn.Module):
    def __init__(
        self, 
        I: torch.Tensor,    # (C, 1, H=N, W=N)
        PSF: torch.Tensor,  # (C, D, H=K, W=K)
    ):
        super().__init__()
        # dimension
        self.D = PSF.shape[1]           # depth
        self.N = I.shape[2]             # image H and W after
        self.K = PSF.shape[2]           # PSF H and W
        self.P = self.K // 2            # padding on each side of I when fft
        self.S = self.N + 2 * self.P    # dim after padding when fft

        # initialize reconstructed volume O
        # NOTE: In haowen's original implementation, he use first channel
        #       as initialization of O. Here we use mean of all channels.
        self.O = (                          # (1, D, H=N, W=N)
            I.mean(dim=0, keepdim=True) / self.D
        ).repeat(1, self.D, 1, 1)

        # we implement convolution by multiplication in Fourier space
        # precompute FFTs of I, PSF, PSFR since they are constant during
        # the RL deconvolution iterations
        I_fft = torch.fft.fftn(             # (C, 1, H=S, W=S)
            I, dim=(-2, -1), s=(self.S, self.S)
        )
        self.PSF_fft = torch.fft.fftn(      # (C, D, H=S, W=S)
            PSF, dim=(-2, -1), s=(self.S, self.S)
        )
        self.PSFR_fft = torch.fft.fftn(     # (C, D, H=S, W=S)
            torch.flip(PSF, dims=[-2, -1]), dim=(-2, -1), s=(self.S, self.S)
        )

        # precompute numerator of RL deconvolution update scheme
        self.numer = torch.fft.ifftn(       # (C, D, H=S, W=S)
            I_fft * self.PSFR_fft, dim=(-2, -1)
        )
        self.numer = torch.sum(             # (1, D, H=S, W=S)
            self.numer, dim=0, keepdim=True
        )
        self.numer = self.numer[            # (1, D, H=N, W=N)
            :, :, self.P:-self.P, self.P:-self.P
        ]
        # for computational stability only
        self.numer = abs(self.numer)

    def forward(self):
        # compute denominator of RL deconvolution update scheme
        # NOTE: In Haowen's original implementation
        #       (https://github.com/hwzhou2020/SVF/blob/main/network.py#L24), 
        #       he mentions there is a weird shift when computing denominator 
        #       and solves by removing padding by 
        #       denom[:, :, 2*self.P:, 2*self.P:] instead of 
        #       denom[:, :, self.P:-self.P, self.P:-self.P]. We find this weird
        #       shift is cause by the padding not remove after first 
        #       convolution and directly use padded result for second 
        #       convolution. We fix this issue by remove padding after first 
        #       convolution and re-pad in second. 
        # first convolve
        O_fft = torch.fft.fftn(             # (1, D, H=S, W=S)
            self.O, dim=(-2,-1), s=(self.S,self.S)
        )
        denom = torch.fft.ifftn(            # (C, D, H=S, W=S)
            O_fft * self.PSF_fft, dim=(-2, -1)
        )
        denom = torch.sum(                  # (C, 1, H=S, W=S)
            denom, dim=1, keepdim=True
        )
        denom = denom[                      # (C, 1, H=N, W=N)
            :, :, self.P:-self.P, self.P:-self.P
        ]
        # second convolve
        denom = torch.fft.fftn(             # (C, 1, H=S, W=S)
            denom, dim=(-2, -1), s=(self.S, self.S)
        ) 
        denom = torch.fft.ifftn(            # (C, D, H=S, W=S)
            denom * self.PSFR_fft, dim=(-2, -1)
        )
        denom = torch.sum(                  # (1, D, H=S, W=S)
            denom, dim=0, keepdim=True
        )
        denom = denom[                      # (1, D, H=N, W=N)
            :, :, self.P:-self.P, self.P:-self.P
        ]
        # for computational stability
        denom = abs(denom)

        self.O = self.O * (self.numer / denom)

        return self.O
