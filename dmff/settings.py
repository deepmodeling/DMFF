from jax.config import config

PRECISION = 'single'  # 'double'

DO_JIT = True

if PRECISION == 'double':
    config.update("jax_enable_x64", True)
    