import torch
import numpy as np

# a = np.arange(10).reshape(5,2)

# b = torch.from_numpy(a)

# print(b)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     x *= x
#     x = x.to("cpu")

x = torch.ones(5, requires_grad=True)

print(x)