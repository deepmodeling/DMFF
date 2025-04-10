import numpy as np
import jax
from jax import jit, value_and_grad, random
import jax.numpy as jnp
from dmff.api import Hamiltonian
from openmm import *
from openmm.app import * 
from openmm.unit import *
from dmff.difftraj import Loss_Generator
import numpy.testing as npt
import pytest


seed = 0
T = 298.15 # temperature
KB = 1.380649e-23
NA = 6.0221408e23
kT = KB * T * NA

m_O = 15.99943
m_H = 1.007947
mass = jnp.tile(jnp.array([m_O, m_H, m_H]), 216)


@pytest.mark.parametrize(
    "pdbfile, prm, values",
    [("tests/data/water_nvt.pdb", "tests/data/qspc-fw.xml", 5412.57173719)])
    # [("tests/data/water_nvt.pdb", "tests/data/qspc-fw.xml", 5401.08336042)])
def test_difftraj(pdbfile, prm, values):

    pdb = PDBFile(pdbfile)
    ff = Hamiltonian(prm)
    pots = ff.createPotential(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.7*nanometer, rigidWater=False)
    efunc = jit(pots.getPotentialFunc())
    params = ff.getParameters().parameters
    cov_map = pots.meta['cov_map']
    box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)

    key = random.PRNGKey(seed)
    state = {}
    state['pos'] = jnp.array(pdb.getPositions()._value).reshape([1, 648, 3])
    state['vel'] = jnp.einsum('ijk,j->ijk', random.normal(key, shape=state['pos'].shape), jnp.sqrt(kT/mass * 1e3)) / 1e3

    @jit
    def L(traj):
        return jnp.sum(traj)

    @jit
    def f_nout(state):
        return jnp.sum(state['pos'])
    metadata = []
    Generator = Loss_Generator(f_nout, box, state['pos'][0], mass, 0.0005, 5, 2, cov_map, 0.7, efunc)
    Loss = Generator.generate_Loss(L, has_aux=False, metadata=metadata)

    v, g = value_and_grad(Loss, argnums=(0, 1))(state, params)
    npt.assert_almost_equal(v, values, decimal=3)   
    print(g) 
