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
    
    @pytest.fixture(scope='class', name='generators')
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
        H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        
        yield H.getGenerators()
        
    def test_ADMPPmeForce_jit(self, generators):
        
        gen = generators[1]
        rc = 4.0
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = jnp.array(pdb.positions._value) * 10
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # neighbor list
        nblist = NeighborList(box, rc)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        pot_pme = gen.getJaxPotential()
        j_pot_pme = jit(value_and_grad(pot_pme))
        
        E_pme, F_pme = j_pot_pme(positions, box, pairs, gen.params)