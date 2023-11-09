import pytest
import jax.numpy as jnp
from jax import jit
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from dmff.common.nblist import NeighborListFreud
from dmff.utils import regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import (
    distribute_v3, 
)
from dmff.admp.spatial import (
    v_pbc_shift, 
)
import freud

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


        nbobj = NeighborList(box, rc, potential.meta["cov_map"])
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
        assert pairs.shape[0] == int(15 * 1.3)
        
    def test_scaled_pairs(self, nblist):
        scaled = pair_buffer_scales(nblist.pairs)
        assert scaled.sum() == 15


class TestFreudNeighborlist:

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
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = np.array([a._value, b._value, c._value]) * 10
        positions = np.array(pdb.positions._value) * 10
        nbobj = NeighborListFreud(box, rc, potential.meta["cov_map"])
        nbobj.capacity_multiplier = 1
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
        
    def test_scaled_pairs(self, nblist):
        
        pairs = nblist.scaled_pairs
        scaled = pair_buffer_scales(nblist.pairs)
        assert pairs.shape[0] == scaled.sum()
