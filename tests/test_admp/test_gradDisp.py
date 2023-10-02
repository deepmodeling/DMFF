import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from jax import jit, value_and_grad
from jax.config import config

config.update("jax_debug_nans", True)

class TestGradDispersion:

    @pytest.mark.parametrize(
        "pdb, prm, values", 
        [
            (
                "tests/data/peg2.pdb", 
                "tests/data/peg.xml", 
                # "peg2.pdb", 
                # "peg.xml", 
                jnp.array(
                [[-9.3132257e-10,  4.0017767e-11,  4.6566129e-10],
                 [ 3.7289283e-11,  1.4551915e-11,  9.0949470e-13],
                 [ 6.9849193e-10, -1.0913936e-11, -1.1641532e-10]]
                )
            ),
        ]
    )
    def test_admp_slater(self, pdb, prm, values):
        pdb = app.PDBFile(pdb)
        H = Hamiltonian(prm) 
        rc = 1.5
        pots = H.createPotential(
            pdb.topology, 
            nonbondedCutoff=rc*unit.nanometer, 
            nonbondedMethod=app.CutoffPeriodic, 
            ethresh=1e-4)

        pot_disp = pots.dmff_potentials['ADMPDispPmeForce']
        
        params = H.getParameters()

        # init positions used to set up neighbor list
        pos = jnp.array(pdb.positions._value)
        n_atoms = len(pos)
        box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)

        # nn list initial allocation
        nbl = NeighborList(box, rc, H.getGenerators()[0].covalent_map)
        nbl.allocate(pos)
        pairs = nbl.pairs

        calc_disp = value_and_grad(pot_disp,argnums=(0,1))
        E, (F, V) = calc_disp(pos, box, pairs, params)
