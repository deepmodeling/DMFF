import pytest
import numpy as np
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff import Hamiltonian, NeighborList


class TestFreeEnergy:

    @pytest.mark.parametrize(
        "pdb, prm, lambdas, energies, dvdls",
        [
            (
                "tests/data/methane_water.pdb", 
                "tests/data/methane_water_coul.xml",
                [0.0, 0.5, 1.0],
                [-3.540451, -3.555031, -3.569612],
                [-0.029161, -0.029161, -0.029161]
            )
        ]
    )
    def test_coul(self, pdb, prm, lambdas, energies, dvdls):
        rcut = 0.5 # nanometers
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(
            pdb.topology, 
            nonbondedMethod=app.PME, 
            constraints=app.HBonds, 
            removeCMMotion=False, 
            nonbondedCutoff=rcut * unit.nanometers,
            useDispersionCorrection=True,
            isFreeEnergy=True,
            coupleIndex=[0, 1, 2, 3, 4],
            coulSoftCore=False,
            vdwSoftCore=False,
            PmeCoeffMethod="gromacs",
            PmeSpacing=0.10
        )
        positions = jnp.array(
            pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        box = jnp.array([
            [ 1.20,  0.00,  0.00],
            [ 0.00,  1.20,  0.00],
            [ 0.00,  0.00,  1.20]
        ], dtype=jnp.float64)
        nbList = NeighborList(box, rc=rcut)
        nbList.allocate(positions)
        pairs = nbList.pairs
        func = jax.value_and_grad(potential.dmff_potentials["NonbondedForce"], argnums=-1)
        for i in range(len(lambdas)):
            ene, dvdl = func(
                positions, 
                box, 
                pairs,
                h.paramtree,
                0.0,
                lambdas[i]
            )
            assert np.allclose(ene, energies[i], atol=1e-2)
            assert np.allclose(dvdl, dvdls[i], atol=1e-2)


    @pytest.mark.parametrize(
        "pdb, prm, lambdas, energies, dvdls",
        [
            (
                "tests/data/methane_water.pdb", 
                "tests/data/methane_water_vdw.xml",
                [0.0, 0.5, 1.0],
                [-0.561297, -0.417853, -0.274409],
                [0.286888, 0.286888, 0.286888]
            )
        ]
    )
    def test_vdw(self, pdb, prm, lambdas, energies, dvdls):
        rcut = 0.5 # nanometers
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(
            pdb.topology, 
            nonbondedMethod=app.PME, 
            constraints=app.HBonds, 
            removeCMMotion=False, 
            nonbondedCutoff=rcut * unit.nanometers,
            useDispersionCorrection=True,
            isFreeEnergy=True,
            coupleIndex=[0, 1, 2, 3, 4],
            coulSoftCore=False,
            vdwSoftCore=False,
            PmeCoeffMethod="gromacs",
            PmeSpacing=0.10
        )
        positions = jnp.array(
            pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        box = jnp.array([
            [ 1.20,  0.00,  0.00],
            [ 0.00,  1.20,  0.00],
            [ 0.00,  0.00,  1.20]
        ], dtype=jnp.float64)
        nbList = NeighborList(box, rc=rcut)
        nbList.allocate(positions)
        pairs = nbList.pairs
        func = jax.value_and_grad(potential.dmff_potentials["NonbondedForce"], argnums=-2)
        for i in range(len(lambdas)):
            ene, dvdl = func(
                positions, 
                box, 
                pairs,
                h.paramtree,
                lambdas[i],
                0.0
            )
            assert np.allclose(ene, energies[i], atol=1e-3)
            assert np.allclose(dvdl, dvdls[i], atol=1e-3)