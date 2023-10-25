import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from jax import jit, value_and_grad

class TestADMPAPI:
    
    """ Test EANN related generators
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
        H = Hamiltonian('tests/data/peg_eann.xml')
        pdb = app.PDBFile('tests/data/peg4.pdb')
        potential = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        
        yield potential, H.paramset

    def test_EANN_energy(self, pot_prm):
        potential, paramset = pot_prm
        rc = 0.4
        pdb = app.PDBFile('tests/data/peg4.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        covalent_map = potential.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential.getPotentialFunc(names=["EANNForce"])
        energy = pot(positions, box, pairs, paramset)
        print(energy)
        np.testing.assert_almost_equal(energy, -0.09797672247940442, decimal=4)

        
