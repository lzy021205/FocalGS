import torch
import torch.nn as nn

class GTnet(nn.Module):

    def __init__(self, in_dim=38, width=64):
        """
        Physics-based camera parameter network.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2)
        ).to("cuda")


        nn.init.xavier_normal_(self.mlp[-1].weight, 0.01)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
            
    def forward(self, camera_state):
        """
        Args:
            camera_state: [1, 38] containing the focal feature, camera position, and viewing direction.
        """
        phys_params = self.mlp(camera_state)
        return phys_params