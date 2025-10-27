import keops_bridge

# exemple avec des tenseurs PyTorch
import torch

x = torch.randn(10, 3)
y = torch.randn(10, 3)

# appel de la fonction C++
result = keops_bridge.gaussian_conv(x, y)

print(result)
