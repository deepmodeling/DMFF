import pytest
import jax.numpy as jnp
from jax import jit
from dmff import NeighborList
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList

class TestNeighborList:
    
    @pytest.fixture(scope="class", name='nblist')
    def test_nblist_init(self):

        """load generators from XML file

        Yields:
            Tuple: (
                ADMPDispForce,
                ADMPPmeForce, # polarized
            )
        """
        rc = 4.0
        H = Hamiltonian('tests/data/admp.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        potential = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        generators = H.getGenerators()
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = np.array([a._value, b._value, c._value]) * 10
        positions = np.array(pdb.positions._value) * 10


        nbobj = NeighborList(box, rc, generators[1].covalent_map)
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
        assert pairs.shape == (15, 3)
        
    def test_pair_mask(self, nblist):
        
        pair, mask = nblist.pair_mask
        assert mask.shape == (15, )
        
    def test_dr(self, nblist):
        
        dr = nblist.dr
        assert dr.shape == (15, 3)
        
    def test_distance(self, nblist):
        
        assert nblist.distance.shape == (15, )

    def test_jit_update(self, nblist):

        positions = jnp.array([
            [12.434,   3.404,   1.540],
            [13.030,   2.664,   1.322],
            [12.312,   3.814,   0.660],
            [14.216,   1.424,   1.103],
            [14.246,   1.144,   2.054],
            [15.155,   1.542,   0.910]
        ])   

        jit(nblist.update)(positions) # pass 
        jit(nblist.update)(positions) # pass 
