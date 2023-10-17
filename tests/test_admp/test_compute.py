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
        rc = 0.4
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        
        covalent_map = potential.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential.getPotentialFunc(names=["ADMPPmeForce"])
        energy = pot(positions, box, pairs, paramset)
        print(energy)
        np.testing.assert_almost_equal(energy, -35.71585296268245, decimal=1)

        
    def test_ADMPPmeForce_jit(self, pot_prm):
        potential, paramset = pot_prm
        rc = 0.4
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        covalent_map = potential.meta["cov_map"]
        # neighbor list
        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        pot = potential.getPotentialFunc(names=["ADMPPmeForce"])
        j_pot_pme = jit(value_and_grad(pot))
        energy, grad = j_pot_pme(positions, box, pairs, paramset.parameters)
        print(energy)
        np.testing.assert_almost_equal(energy, -35.71585296268245, decimal=1)
