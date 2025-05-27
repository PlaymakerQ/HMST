import torch
import torch.nn as nn
from manifolds.lorentz_functions import givens_rotations

class Rotation(nn.Module):

    def __init__(self, num_dim, random_init=True):
        super(Rotation, self).__init__()
        self.num_dim = num_dim
        init_weights = torch.zeros(1, self.num_dim, dtype=torch.float32)
        if random_init:
            num_angles = num_dim // 2
            init_angles = torch.randn(1, num_angles) * torch.pi
            init_weights[:, ::2] = torch.cos(init_angles)
            init_weights[:, 1::2] = torch.sin(init_angles)
        else:
            init_weights[:, ::2] = 1.0
            init_weights[:, 1::2] = 0.0
        self.rotary_weights = nn.Parameter(init_weights)

    def forward(self, input_embeddings):
        shape = input_embeddings.size()
        dim = shape[-1]
        input_embeddings = input_embeddings.reshape(-1, dim)
        if dim != self.num_dim:  # rotation for the Lorentz model
            assert (dim - self.num_dim) == 1
            time_axis = input_embeddings[:, 0:1]
            input_embeddings = input_embeddings[:, 1:]
            rotary_weights = self.rotary_weights.expand(input_embeddings.size())
            rotated_embeddings = givens_rotations(rotary_weights, input_embeddings)
            rotated_embeddings = torch.cat([time_axis, rotated_embeddings], dim=-1)
            rotated_embeddings = rotated_embeddings.reshape(shape)
            return rotated_embeddings
        else:  # rotation for the other model
            rotary_weights = self.rotary_weights.expand(input_embeddings.size())
            rotated_embeddings = givens_rotations(rotary_weights, input_embeddings)
            rotated_embeddings = rotated_embeddings.reshape(shape)
            return rotated_embeddings





if __name__ == '__main__':

    embs = torch.randn(3, 4, 4)
    model = Rotation(4, random_init=False)
    model.train()
    outout = model(embs)