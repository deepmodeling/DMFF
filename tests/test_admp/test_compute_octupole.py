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
        H1 = Hamiltonian('tests/data/admp_mono.xml')
        H2 = Hamiltonian('tests/data/admp_nonpol.xml')
        H3 = Hamiltonian('tests/data/admp_octupole.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        potential = H.createPotential(pdb.topology, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        potential_aux = H.createPotential(pdb.topology, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5, has_aux=True)
        potential1 = H1.createPotential(pdb.topology, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        potential2 = H2.createPotential(pdb.topology, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        potential3 = H3.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5, has_aux=True)
        
        yield potential, potential_aux, potential1, potential2, potential3, H.paramset, H1.paramset, H2.paramset, H3.paramset

    def test_ADMPPmeForce_octupole(self, pot_prm):
        potential, potential_aux, potential1, potential2, potential3, paramset, paramset1, paramset2, paramset3 = pot_prm
        rc = 0.4
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        
        covalent_map = potential3.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential3.getPotentialFunc(names=["ADMPPmeForce"])

        aux = dict()
        U_ind = jnp.zeros((6, 3))
        aux["U_ind"] = U_ind
        
        energy_and_aux = pot(positions, box, pairs, paramset3, aux)
        energy = energy_and_aux[0]
        print("Octupole Included Energy: ", energy)
        np.testing.assert_almost_equal(energy, -36.32748562120901, decimal=1)
        
    def test_ADMPPmeForce_octupole_water_216(self):
        rc = 8.0
        H3 = Hamiltonian('tests/data/admp_octupole.xml')
        pdb = app.PDBFile('tests/data/water_216.pdb')
        potential3 = H3.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5, has_aux=True)
        paramset3 = H3.paramset
        
        rc = 0.8
        pdb = app.PDBFile('tests/data/water_216.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        
        covalent_map = potential3.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential3.getPotentialFunc(names=["ADMPPmeForce"])

        aux = dict()
        U_ind = jnp.zeros((216 * 3, 3))
        aux["U_ind"] = U_ind
        
        energy_and_aux = pot(positions, box, pairs, paramset3, aux)
        energy = energy_and_aux[0]
        print("Octupole Included Energy: ", energy)
        np.testing.assert_almost_equal(energy, -9294.40, decimal=1)
    
    def test_ADMPPmeForce_octupole_ala5(self):
        rc = 12.0
        H3 = Hamiltonian('tests/data/ala5_dmff.xml')
        pdb = app.PDBFile('tests/data/ala5_mpid.pdb')
        potential3 = H3.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, has_aux=True)
        paramset3 = H3.paramset
        
        rc = 1.2
        pdb = app.PDBFile('tests/data/ala5_mpid.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        
        covalent_map = potential3.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential3.getPotentialFunc(names=["ADMPPmeForce"])

        aux = dict()
        U_ind = jnp.zeros((54, 3))
        aux["U_ind"] = U_ind
        
        energy_and_aux = pot(positions, box, pairs, paramset3, aux)
        energy = energy_and_aux[0]
        print("Octupole Included Energy: ", energy)
        np.testing.assert_almost_equal(energy, 63.2, decimal=1)
