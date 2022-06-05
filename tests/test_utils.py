import jax.numpy as jnp
import pytest
from dmff.utils import regularize_pairs, pair_buffer_scales
import numpy.testing as npt

class TestUtils:
    
    @pytest.fixture(scope='class', name='pairs')
    def test_init_pair(self):
    
        pairs = jnp.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ], dtype=int)
        yield pairs
    
    def test_regularize_pairs(self, pairs):
        
        ans = regularize_pairs(pairs)
        npt.assert_array_equal(ans, jnp.array([
            [-1, -2],
            [0, -2],
            [0,  1],
            [0, -1]
        ]))
        
    def test_pair_buffer_scales(self, pairs):
        
        ans = pair_buffer_scales(pairs)
        npt.assert_array_equal(ans, jnp.array([0, 0, 1, 0]))