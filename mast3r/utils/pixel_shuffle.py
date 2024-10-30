import numpy as np
from tinygrad.tensor import Tensor

def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    # Check input shape
    n, c_r2, h, w = input.shape
    assert c_r2 % (upscale_factor ** 2) == 0, "C must be divisible by r^2"
    
    c = c_r2 // (upscale_factor ** 2)

    # Reshape the input tensor
    input_reshaped = input.reshape(n, c, upscale_factor, upscale_factor, h, w)
    
    # Rearrange the tensor
    output = input_reshaped.permute(0, 1, 4, 2, 5, 3).reshape(n, c, h * upscale_factor, w * upscale_factor)
    
    return output