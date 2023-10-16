import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from jax import jit, value_and_grad

class TestADMPAPI:
    
    """ Test ADMP related generators
    """
    
    @pytest.fixture(scope='class', name='pot_prm')
    def test_init(self):
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
        
        yield potential, H.paramset

    def test_ADMPPmeForce(self, pot_prm):
        potential, paramset = pot_prm
        rc = 4.0
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = np.array(pdb.positions._value) * 10
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = np.array([a._value, b._value, c._value]) * 10
        # neighbor list
        
        covalent_map = potential.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential.getPotentialFunc()
        energy = pot(positions, box, pairs, paramset)

        
    def test_ADMPPmeForce_jit(self, pot_prm):
        potential, paramset = pot_prm
        rc = 4.0
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = jnp.array(pdb.positions._value) * 10
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        covalent_map = potential.meta["cov_map"]
        # neighbor list
        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        pot = potential.getPotentialFunc()
        j_pot_pme = jit(value_and_grad(pot))
        energy = j_pot_pme(positions, box, pairs, paramset.parameters)
