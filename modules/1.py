import torch
import torch.nn as nn
import math

a = torch.ones(1, 2, 1) / 2.0
b = torch.ones(1, 2, 1) / 2.0

loss = nn.CrossEntropyLoss()
output = loss(a, b)
print(output, 2.0 * math.log(2))