from dmff.settings import DO_JIT
from jax import jit, vmap
import jax.numpy as jnp

def jit_condition(*args, **kwargs):
    def jit_deco(func):
        if DO_JIT:
            return jit(func, *args, **kwargs)
        else:
            return func
    return jit_deco


@jit_condition()
@vmap
def regularize_pairs(p):
    dp = p[1] - p[0]
    dp = jnp.piecewise(dp, (dp<=0, dp>0), (lambda x: jnp.array(-1), lambda x: jnp.array(0)))
    p += dp
    return p


@jit_condition()
@vmap
def pair_buffer_scales(p):
    return jnp.piecewise(
            p[0] - p[1], 
            (p[0] - p[1] < 0, p[0] - p[1] == 0), 
            (lambda x: jnp.array(1), lambda x: jnp.array(0)))
