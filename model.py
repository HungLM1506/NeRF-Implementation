import torch
import torch.nn as nn
from utils import positional_encoding


class NeRF_Base(nn.Module):
    def __init__(self):
        super(NeRF_Base, self).__init__()

        # Initial input linear layers for xyz
        self.fc1_block1 = nn.Linear(2 * 3 * 10 + 3, 256)
        self.fc2_block1 = nn.Linear(256, 256)
        self.fc3_block1 = nn.Linear(256, 256)
        self.fc4_block1 = nn.Linear(256, 256)

        # Linear layers for ray direction
        self.fc1_d = nn.Linear(2 * 3 * 4 + 3, 256)

        # Linear layers after concatenation
        self.fc1_block2 = nn.Linear(256 + 2 * 3 * 10 + 3, 256)
        self.fc2_block2 = nn.Linear(256, 256)
        self.fc3_block2 = nn.Linear(256, 256)
        self.fc4_block2 = nn.Linear(256, 256)

        # Output layers
        self.linear_density = nn.Linear(256, 1)

        # Linear layers for RGB
        self.fc1_block3 = nn.Linear(256, 256)
        self.fc2_block3 = nn.Linear(256 + 2 * 3 * 4 + 3, 128)

        self.linear_rgb = nn.Linear(128, 3)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, r_d):
        # Positional encoding
        x_encoded = positional_encoding(x, L=10)  # [64, 10000, 63]
        r_d_encoded = positional_encoding(r_d, L=4)  # [10000, 27]

        # Process x through initial layers
        x = self.relu(self.fc1_block1(x_encoded))
        x = self.relu(self.fc2_block1(x))
        x = self.relu(self.fc3_block1(x))
        x = self.relu(self.fc4_block1(x))

        # concat x again
        x = torch.cat([x, x_encoded], dim=-1)

        x = self.relu(self.fc1_block2(x))
        x = self.relu(self.fc2_block2(x))
        x = self.relu(self.fc3_block2(x))
        x = self.fc4_block2(x)

        # output density
        density = self.relu(self.linear_density(x))

        # Process ray direction
        x = self.fc1_block3(x)

        x = torch.cat([x, r_d_encoded], dim=-1)
        # Process after concatenation
        x = self.relu(self.fc2_block3(x))
        rgb = self.linear_rgb(x)
        rgb = self.sigmoid(rgb)

        return rgb, density
