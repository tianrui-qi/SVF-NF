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

        self.I = I                          # (C, 1, H=N, W=N)
        self.PSF = PSF                      # (C, D, H=K, W=K)
        self.PSFR = torch.flip(             # (C, D, H=K, W=K)
            PSF, dims=[-2, -1]
        )
        # initialize reconstructed volume O
        # NOTE: In Haowen's implementation, he use first channel as 
        #       initialization of O. Here we use mean of all channels.
        self.O = (                          # (1, D, H=N, W=N)
            I.mean(dim=0, keepdim=True) / self.D
        ).repeat(1, self.D, 1, 1)

        # implement convolution by multiplication in Fourier space
        # NOTE: In Haowen's implementation, he precompute FFT of I, PSF, PSFR.
        #       However, that would require lots of memory, expecially for 
        #       PSF and PSFR. Thus, in our implementation, we will compute FFT 
        #       on the fly.

        # precompute numerator of RL deconvolution update scheme
        self.numer = torch.fft.ifftn(       # (C, D, H=S, W=S)
            self.fft(self.I) * self.fft(self.PSFR), dim=(-2, -1)
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
        # NOTE: In Haowen's implementation
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
        denom = torch.fft.ifftn(            # (C, D, H=S, W=S)
            self.fft(self.O) * self.fft(self.PSF), dim=(-2, -1)
        )
        denom = torch.sum(                  # (C, 1, H=S, W=S)
            denom, dim=1, keepdim=True
        )
        denom = denom[                      # (C, 1, H=N, W=N)
            :, :, self.P:-self.P, self.P:-self.P
        ]
        # second convolve
        denom = torch.fft.ifftn(            # (C, D, H=S, W=S)
            self.fft(denom) * self.fft(self.PSFR), dim=(-2, -1)
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

    def fft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(x, dim=(-2, -1), s=(self.S, self.S))
