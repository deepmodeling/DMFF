from jax.config import config

PRECISION = 'float'  # 'double'

DO_JIT = False

DEBUG = False

if PRECISION == 'double':
    config.update("jax_enable_x64", True)
    
__all__ = ['PRECISION', 'DO_JIT', 'DEBUG']
