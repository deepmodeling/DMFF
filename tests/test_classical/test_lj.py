import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian
import pytest
from dmff import NeighborList


class TestVdW:
    
    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj2.pdb", "tests/data/lj2.xml", -1.85001802444458)])
    def test_lj_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        gen = h.getGenerators()[0]
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        nblist = NeighborList(box, 4.0, gen.covalent_map)
        nblist.allocate(pos)
        pairs = nblist.pairs
        ljE = potential.getPotentialFunc()
        energy = ljE(pos, box, pairs, h.paramtree)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(ljE)(pos, box, pairs, h.paramtree)
        npt.assert_almost_equal(energy, value, decimal=3)
        

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj3.pdb", "tests/data/lj3.xml", -2.001220464706421)])
    def test_lj_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        gen = h.getGenerators()[0]
        nblist = NeighborList(box, 4.0, gen.covalent_map)
        nblist.allocate(pos)
        pairs = nblist.pairs
        ljE = potential.getPotentialFunc()
        energy = ljE(pos, box, pairs, h.paramtree)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(ljE)(pos, box, pairs, h.paramtree)
        npt.assert_almost_equal(energy, value, decimal=3)
        
    def test_lj_params_check(self):
        pdb = app.PDBFile("tests/data/lj3.pdb")
        h = Hamiltonian("tests/data/lj3.xml")
        potential = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])     
        gen = h.getGenerators()[0]
        nblist = NeighborList(box, 4.0, gen.covalent_map)
        nblist.allocate(pos)              
        pairs = nblist.pairs 
        ljE = potential.getPotentialFunc()
        with pytest.raises(TypeError):
            energy = ljE(pos, box, pairs, h.getGenerators()[0].paramtree)
            
        energy = jax.jit(ljE)(pos, box, pairs, h.paramtree)  # jit will optimized away type check
        force = jax.grad(jax.jit(ljE))(pos, box, pairs, h.paramtree)