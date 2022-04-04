import jax.numpy as jnp
import numpy as np
from jax import jit, grad
import jax


jax.profiler.start_trace('./tensorboard/')
def affine(x, w, b):
    return x * w + b

x = jnp.ones((5, ))
w = 3
b = 2

j = jit(affine)(x, w, b)
j.block_until_ready()

jax.profiler.stop_trace()