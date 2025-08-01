import torch
import numpy as np
from typing import List

data: List = [
    [
        [[0.5020, 0.5020, 0.5020], [0.5098, 0.5098, 0.5137], [0.5098, 0.5059, 0.5137]],
        [[0.3686, 1.0000, 0.4039], [0.3686, 1.0000, 0.4039], [0.3686, 1.0000, 0.4039]],
        [[0.3686, 1.0000, 0.4039], [0.3686, 1.0000, 0.4039], [0.3686, 1.0000, 0.4039]],
        [[0.3686, 1.0000, 0.4039], [0.3686, 1.0000, 0.4039], [0.3686, 1.0000, 0.4039]],
    ]
]
tensor = torch.tensor(data)
print(tensor.shape)

for item in tensor:
    print(item)
    print(item.shape)
