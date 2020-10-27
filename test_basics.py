
import torch
from torch import nn

# assume square dimensions
size = 1200

layer = nn.Linear(size*size, 100)

# generate a test image tensor
# Torch uses NCHW
image_tensor = torch.randn(1000, 1200, 1200)

result = layer(image_tensor.view(image_tensor.size(0), -1))

print(result.shape)

