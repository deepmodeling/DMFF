import pytest
import jax.numpy as jnp
from dmff import NeighborList

class TestNeighborList:
    
    @pytest.fixture(scope="class", name='nblist')
    def test_nblist_init(self):
        positions = jnp.array([
            [12.434,   3.404,   1.540],
            [13.030,   2.664,   1.322],
            [12.312,   3.814,   0.660],
            [14.216,   1.424,   1.103],
            [14.246,   1.144,   2.054],
            [15.155,   1.542,   0.910]
        ])
        box = jnp.array([31.289,   31.289,   31.289])
        r_cutoff = 4.0
        nbobj = NeighborList(box, r_cutoff)
        nbobj.allocate(positions)
        yield nbobj
        
    def test_update(self, nblist):

        positions = jnp.array([
            [12.434,   3.404,   1.540],
            [13.030,   2.664,   1.322],
            [12.312,   3.814,   0.660],
            [14.216,   1.424,   1.103],
            [14.246,   1.144,   2.054],
            [15.155,   1.542,   0.910]
        ])
        nblist.update(positions)
        
    def test_pairs(self, nblist):
        
        pairs = nblist.pairs
        assert pairs.shape == (18, 2)
        
    def test_pair_mask(self, nblist):
        
        pair, mask = nblist.pair_mask
        assert mask.shape == (18, )
        
    def test_dr(self, nblist):
        
        dr = nblist.dr
        assert dr.shape == (18, 3)
        
    def test_distance(self, nblist):
        
        assert nblist.distance.shape == (18, )
