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
        potential3 = H3.createPotential(pdb.topology, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5, has_aux=True)
        
        yield potential, potential_aux, potential1, potential2, potential3, H.paramset, H1.paramset, H2.paramset, H3.paramset

    def test_ADMPPmeForce(self, pot_prm):
        potential, potential_aux, potential1, potential2, potential3, paramset, paramset1, paramset2, paramset3 = pot_prm
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
        potential, potential_aux, potential1, potential2, potential3, paramset, paramset1, paramset2, paramset3 = pot_prm
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
        print('hahahah', energy)
        np.testing.assert_almost_equal(energy, -35.71585296268245, decimal=1)

    def test_ADMPPmeForce_aux(self, pot_prm):
        potential, potential_aux, potential1, potential2, potential3, paramset, paramset1, paramset2, paramset3 = pot_prm
        rc = 0.4
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        covalent_map = potential.meta["cov_map"]

        # check map-atomtype & map-poltype
        print('map-atomtype', potential.meta["ADMPPmeForce_map_atomtype"])
        print("map-poltype", potential.meta["ADMPPmeForce_map_poltype"])
        # neighbor list
        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        
        aux = {
            "U_ind": jnp.zeros((len(positions),3)),
        }
        pot = potential_aux.getPotentialFunc(names=["ADMPPmeForce"])
        j_pot_pme = jit(value_and_grad(pot, has_aux=True))
        (energy, grad), aux = j_pot_pme(positions, box, pairs, paramset.parameters, aux=aux)
        print('hahahah', energy)
        np.testing.assert_almost_equal(energy, -35.71585296268245, decimal=1)
   

    def test_ADMPPmeForce_mono(self, pot_prm):
        potential, potential_aux, potential1, potential2, potential3, paramset, paramset1, paramset2, paramset3 = pot_prm
        rc = 0.4
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        
        covalent_map = potential1.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential1.getPotentialFunc(names=["ADMPPmeForce"])
        energy = pot(positions, box, pairs, paramset1)
        print(energy)
        np.testing.assert_almost_equal(energy, -66.46778622510325, decimal=2)
    

    def test_ADMPPmeForce_nonpol(self, pot_prm):
        potential, potential_aux, potential1, potential2, potential3, paramset, paramset1, paramset2, paramset3 = pot_prm
        rc = 0.4
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions = jnp.array(positions)
        a, b, c = pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        box = jnp.array([a, b, c])
        # neighbor list
        
        covalent_map = potential2.meta["cov_map"]

        nblist = NeighborList(box, rc, covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        pot = potential2.getPotentialFunc(names=["ADMPPmeForce"])
        energy = pot(positions, box, pairs, paramset2)
        print(energy)
        np.testing.assert_almost_equal(energy, -31.65932348, decimal=2)

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