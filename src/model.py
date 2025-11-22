import torch


class Render(torch.nn.Module):
    def __init__(
        self, 
        Q: int = 32, out_dim: int = 1, 
        hidden_dim: int = 32, num_layers: int = 2, 
        layernorm: bool = False
    ) -> None:
        super().__init__()
        layers = [
            # input layer
            torch.nn.Linear(Q, hidden_dim),
            torch.nn.LayerNorm(hidden_dim) 
            if layernorm else torch.nn.Identity(),
            torch.nn.ReLU(),
            # insert more hidden layers by for loop
            # output layer
            torch.nn.Linear(hidden_dim, out_dim)
        ]
        for _ in range(num_layers - 1):
            layers[-1:-1] = [
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim) 
                if layernorm else torch.nn.Identity(),
                torch.nn.ReLU(),
            ]
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (..., Q) -> (..., out_dim)


class FeatureXY(torch.nn.Module):
    x0: torch.Tensor
    y0: torch.Tensor
    x1: torch.Tensor
    y1: torch.Tensor
    wx: torch.Tensor
    wy: torch.Tensor

    def __init__(
        self, X: int, Y: int, Q: int = 32, downsample: int = 1
    ) -> None:
        super().__init__()
        # dimension
        Xd = X // downsample
        Yd = Y // downsample

        # NOTE:
        # when representing a coordinate pair, use (x, y)
        # when indexing a tensor, use [y, x]

        # learnable feature tensor M(x, y, :)
        self.M = torch.nn.Parameter(2e-4 * torch.rand((Yd, Xd, Q)) - 1e-4)

        # bilinear interpolation
        # generate normalized coordinates in (0, 1)
        xs = torch.linspace(0.5 / X, 1 - 0.5 / X, X)        # (X, )
        ys = torch.linspace(0.5 / Y, 1 - 0.5 / Y, Y)        # (Y, )
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")    # (X, Y), (X, Y)
        xy = torch.stack((xv.flatten(), yv.flatten())).t()  # (X*Y, 2)
        # scale coordinates to downsampled resolution in (0, Xd-1), (0, Yd-1)
        xy = xy * xy.new_tensor([Xd, Yd])                   # (X*Y, 2)
        # split into integer (grid indices) and fractional (bilinear weights)
        integer = xy.long()
        decimal = xy - integer.float()
        # precomputed interpolation indices
        self.register_buffer("x0", integer[:, 0].clamp(min=0, max=Xd - 1))
        self.register_buffer("y0", integer[:, 1].clamp(min=0, max=Yd - 1))
        self.register_buffer("x1", (self.x0 + 1).clamp(max=Xd - 1))
        self.register_buffer("y1", (self.y0 + 1).clamp(max=Yd - 1))
        # bilinear interpolation weights
        self.register_buffer("wx", decimal[:, 0:1]) # fractional offset along x
        self.register_buffer("wy", decimal[:, 1:2]) # fractional offset along y

    def forward(self) -> torch.Tensor:
        return (    # (Y, X, Q)
            self.M[self.y0, self.x0] * (1.0 - self.wx) * (1.0 - self.wy) +
            self.M[self.y0, self.x1] * self.wx * (1.0 - self.wy) +
            self.M[self.y1, self.x0] * (1.0 - self.wx) * self.wy +
            self.M[self.y1, self.x1] * self.wx * self.wy 
        )


class RenderXY(torch.nn.Module):
    def __init__(
        self, 
        X: int, Y: int, Q: int = 32, downsample: int = 1,
        hidden_dim: int = 32, num_layers: int = 2, layernorm: bool = False
    ) -> None:
        super().__init__()
        self.M = FeatureXY(X, Y, Q=Q, downsample=downsample)
        self.mlp = Render(
            Q=Q,
            hidden_dim=hidden_dim, num_layers=num_layers, layernorm=layernorm
        )

    def forward(self):
        x = self.M()        # (Y, X, Q)
        x = self.mlp(x)     # (Y, X, 1)
        return x
