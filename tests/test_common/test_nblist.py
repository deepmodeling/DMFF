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
import jax_md

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
        
    def test_scaled_pairs(self, nblist):
        
        pairs = nblist.scaled_pairs
        scaled = pair_buffer_scales(nblist.pairs)
        assert pairs.shape[0] == scaled.sum()

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

class TestFreudNeighborlist:

    # @pytest.fixture(scope='class', name='freud_nblist')
    def test_raw_freud(self):

        system = freud.data.make_random_system(box_size=10, num_points=10)
        box = system[0].to_matrix()
        points = system[1]
        rc = 4.0
        freud_nblist = freud.AABBQuery(box, points).query(points, {'r_max': rc, 'exclude_ii':True}).toNeighborList()
        
        displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
        jax_md_nblist_fn = jax_md.partition.neighbor_list(displacement_fn, box, rc, 0, format=jax_md.partition.OrderedSparse)
        jax_md_nblist = jax_md_nblist_fn.allocate(points)

        jax_pairs = jax_md_nblist.idx.T
        freud_pairs = freud_nblist[:]
        freud_mask = freud_pairs[:, 0] < freud_pairs[:, 1]
        freud_pairs = freud_pairs[freud_mask]

        jax_pairs = regularize_pairs(jax_pairs)
        buffer_scales = pair_buffer_scales(jax_pairs).astype(bool)
        jax_pairs = jax_pairs[buffer_scales]
        box_inv = jnp.linalg.inv(box)
        r1 = distribute_v3(points, jax_pairs[:, 0])
        r2 = distribute_v3(points, jax_pairs[:, 1])
        dr = r1 - r2
        dr = v_pbc_shift(dr, box, box_inv)
        jax_md_distance = jnp.linalg.norm(dr, axis=-1)

        assert len(jax_pairs) == len(freud_pairs)

        freud_distance = freud_nblist.distances[freud_mask]

        assert len(jax_md_distance) == len(freud_distance)
        npt.assert_allclose(sorted(jax_md_distance), sorted(freud_distance), rtol=1e-3)

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
        nbobj = NeighborListFreud(box, rc, generators[1].covalent_map)
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
