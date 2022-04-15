from jax.config import config

PRECISION = 'double'  # 'double'

DO_JIT = True

if PRECISION == 'double':
    config.update("jax_enable_x64", True)
    
