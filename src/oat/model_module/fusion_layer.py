import torch
import torch.nn as nn


class BaseFusionLayer(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()

        self.in_channels = torch.Tensor(in_channels)
        self.out_channels = out_channels


class GatedLinearUnitFusionLayer(BaseFusionLayer):

    def __init__(self, in_channels: list[int],
                 out_channels: int,
                 hidden_channels: int,
                 n_blocks: int = 1,
                 dropout_p: float = 0.0,
                 residual_learning: bool = True
                 ):
        super().__init__(in_channels, out_channels)

        self.residual_learning = residual_learning

        in_dim = int(torch.tensor(self.in_channels).sum().item())


        blocks = []
        for _ in range(n_blocks):
            block = [
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 2*hidden_channels),
                nn.GLU(dim=-1)
            ]
            if dropout_p > 0.0:
                block.append(nn.Dropout(p=dropout_p))

            blocks.append(nn.Sequential(*block))
            in_dim = hidden_channels

        if residual_learning:
            # for residual learning we must be able to iterate the blocks
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.Sequential(*blocks)

        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, *inputs: torch.Tensor):
        x = torch.cat(inputs, dim=-1)

        if self.residual_learning:
            x = self.blocks[0](x)
            for block in self.blocks[1:]:
                # only if there are more than 1 blocks, we there is residual learning
                res = x
                x = block(x)
                x = x + res
        else:
            x = self.blocks(x)

        return self.readout(x)
