from jax import jit, vmap, tree_util
import jax.numpy as jnp
from dmff.settings import DO_JIT


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
    dp = jnp.piecewise(dp, (dp<=0, dp>0), (lambda x: jnp.array(1), lambda x: jnp.array(0)))
    dp_vec = jnp.array([dp, 2*dp])
    p = p - dp_vec
    return p


@jit_condition()
@vmap
def pair_buffer_scales(p):
    return jnp.piecewise(
        p[0] - p[1], 
        (p[0] - p[1] < 0, p[0] - p[1] >= 0), 
        (lambda x: jnp.array(1), lambda x: jnp.array(0))
    )


def isinstance_jnp(*args):
    
    def _check(arg):
        if not isinstance(arg, jnp.ndarray):
            raise TypeError(
                'all arguments must be jnp.array, \
                otherwise they won\'t be able to take derivatives \
                on these variables from outside of potential_fn anyway'
            )
    
    for arg in args:
        tree_util.tree_map(lambda arg: _check(arg), args[0])
