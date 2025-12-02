import torch


class FeatureHW(torch.nn.Module):
    def __init__(
        self, 
        H: int, W: int, downsample: int, 
        Q: int = 32, 
    ) -> None:
        super().__init__()
        # dimension
        self.H = H
        self.W = W
        Hd = H // downsample
        Wd = W // downsample
        # learnable feature tensor M
        self.M = torch.nn.Parameter(2e-4 * torch.rand((Hd, Wd, Q)) - 1e-4)

    def forward(self) -> torch.Tensor:
        m = torch.nn.functional.interpolate(    # (Hd, Wd, Q) -> (H, W, Q)
            self.M.permute(2, 0, 1).unsqueeze(0), 
            size=(self.H, self.W), mode="bilinear",
        ).squeeze(0).permute(1, 2, 0)
        return m    # (H, W, Q)


class FeatureD(torch.nn.Module):
    def __init__(
        self,
        Dd: int, z_min: float, z_max: float,
        Q: int = 32,
    ) -> None:
        super().__init__()
        # dimension
        self.Dd = int(Dd)           # number of depth features
        self.z_min = float(z_min)   # min depth in physical units
        self.z_max = float(z_max)   # max depth in physical units
        # learnable feature tensor U
        self.U = torch.nn.Parameter(5e-1 * torch.randn(self.Dd, Q))

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        # make sure z is on same device / dtype as U
        z = z_grid.to(self.U.device).to(self.U.dtype)
        # clamp to [z_min, z_max] (out-of-range z use boundary embeddings)
        z = z.clamp(self.z_min, self.z_max)
        # map physical z to continuous index in [0, Dd - 1]
        # NOTE: assume z_max > z_min
        z_idx = (z - self.z_min) * (self.Dd - 1) / (self.z_max - self.z_min)
        # left / right indices for linear interpolation
        z0 = torch.floor(z_idx).long()
        z1 = (z0 + 1).clamp(max=self.Dd - 1)
        z0 = z0.clamp(min=0, max=self.Dd - 1)
        # interpolation weight along depth
        w = (z_idx - z0.to(z_idx.dtype)).unsqueeze(-1)  # (D, 1)
        # gather latent depth features and linearly interpolate
        f0 = self.U[z0]                 # (D, Q)
        f1 = self.U[z1]                 # (D, Q)
        u = (1.0 - w) * f0 + w * f1     # (D, Q)
        return u    # (D, Q)


class Render(torch.nn.Module):
    def __init__(
        self, 
        Q: int = 32, out_dim: int = 1, 
        hidden_dim: int = 32, num_layers: int = 2, layernorm: bool = False
    ) -> None:
        super().__init__()
        layers = []
        # input layer
        layers += [
            torch.nn.Linear(Q, hidden_dim),
            torch.nn.LayerNorm(hidden_dim) 
            if layernorm else torch.nn.Identity(),
            torch.nn.ReLU(),
        ]
        # hidden layers
        for _ in range(num_layers - 1):
            layers += [
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim) 
                if layernorm else torch.nn.Identity(),
                torch.nn.ReLU(),
            ]
        # output layer
        layers += [
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.ReLU(),
        ]
        # assemble MLP
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (..., Q) -> (..., out_dim)


class Render2D(torch.nn.Module):
    def __init__(
        self, 
        H: int, W: int, downsample: int, 
        Q: int = 32, 
        hidden_dim: int = 32, num_layers: int = 2, layernorm: bool = False
    ) -> None:
        super().__init__()
        self.M = FeatureHW(H, W, downsample=downsample, Q=Q)
        self.mlp = Render(
            Q=Q,
            hidden_dim=hidden_dim, num_layers=num_layers, layernorm=layernorm
        )

    def forward(self) -> torch.Tensor:
        # feature
        m = self.M()        # (H, W, Q)
        # render
        x = self.mlp(m)     # (H, W, 1)
        x = x.squeeze(-1)   # (H, W)
        return x


class Render3D(torch.nn.Module):
    def __init__(
        self, 
        H: int, W: int, downsample: int, 
        Dd: int, z_min: float, z_max: float,
        Q: int = 32, 
        hidden_dim: int = 32, num_layers: int = 2, layernorm: bool = False
    ) -> None:
        super().__init__()
        self.M = FeatureHW(H=H, W=W, downsample=downsample, Q=Q)
        self.U = FeatureD(Dd=Dd, z_min=z_min, z_max=z_max, Q=Q)
        self.mlp = Render(
            Q=Q,
            hidden_dim=hidden_dim, num_layers=num_layers, layernorm=layernorm
        )

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        # feature
        m = self.M()                # (H, W, Q)
        u = self.U(z_grid)          # (D, Q)
        # broadcast to (D, H, W, Q)
        x = (                       # (D, H, W, Q)
            m.unsqueeze(0) *        # (1, H, W, Q)
            u[:, None, None, :]     # (D, 1, 1, Q)
        )
        # render
        x = self.mlp(x)             # (D, H, W, 1)
        x = x.squeeze(-1)           # (D, H, W)
        return x
