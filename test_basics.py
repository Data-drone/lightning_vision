
import torch
from torch import nn
import torchvision.models as models
resnet18 = models.resnet18()

# assume square dimensions
size = 1200

layer = nn.Linear(size*size, 100)
conv_1 = nn.Conv2d(3, 64, 7, 2, 3)

# generate a test image tensor
# Torch uses NCHW
image_tensor = torch.randn(10, 3, 255, 255)
result = resnet18(image_tensor)

#result = layer(image_tensor.view(image_tensor.size(0), -1))

print(result.shape)

