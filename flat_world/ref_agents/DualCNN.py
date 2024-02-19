import torch
from torch import nn


class _ConvLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(4, 20, (3,), padding=1),
            nn.MaxPool1d((2,)),
            nn.GELU(),
            nn.Conv1d(20, 10, (5,), padding=2),
            nn.MaxPool1d((2,)),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class DualCNN(nn.Module):
    def __init__(self, obs_resolution=20):
        super().__init__()
        self.conv_layers = _ConvLayers()
        conv_out_size = 10 * obs_resolution // 4
        self.final_layers = nn.Sequential(
            nn.LayerNorm(conv_out_size),
            nn.Linear(conv_out_size, 100),
            nn.GELU(),
            nn.Linear(100, 6),
        )

    def forward(self, std_observation):
        x = self.conv_layers(std_observation)
        x = torch.flatten(x, start_dim=1)
        x_output = self.final_layers(x)
        # Returns action logits (1x5)
        motion = x_output[:, :3]
        rot = x_output[:, 3:]
        return motion, rot,
