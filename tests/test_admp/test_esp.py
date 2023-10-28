import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from jax import jit, value_and_grad


def test_admp_esp():
    H = Hamiltonian('tests/data/admp_charge.xml')
    pdb = app.PDBFile('tests/data/water.pdb')
    pdb_probe = app.PDBFile("tests/data/water_ion.pdb")
    potential = H.createPotential(
        pdb.topology, nonbondedMethod=app.NoCutoff, nonbondedCutoff=100*unit.angstrom, ethresh=5e-4, step_pol=1, has_aux=False
    )
    es_pot = H.generators["ADMPPmeForce"].pme_force.generate_esp()
    pot = potential.getPotentialFunc()

    potential_prob = H.createPotential(
        pdb_probe.topology, nonbondedMethod=app.NoCutoff, nonbondedCutoff=100*unit.angstrom, ethresh=5e-4, step_pol=1, has_aux=False
    )
    pot_prob = potential_prob.getPotentialFunc()



    positions = jnp.array(pdb.positions.value_in_unit(unit.nanometer))
    positions_prob = jnp.array(pdb_probe.positions.value_in_unit(unit.nanometer))

    box = jnp.array([
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 100.0]
    ])
    pairs = []
    cov_map = potential.meta["cov_map"]
    for ii in range(3):
        for jj in range(ii+1, 3):
            pairs.append([ii, jj, cov_map[ii, jj]])
    pairs = jnp.array(pairs)

    pairs_prob = []
    cov_map = potential_prob.meta["cov_map"]
    for ii in range(4):
        for jj in range(ii+1, 4):
            pairs_prob.append([ii, jj, cov_map[ii, jj]])
    pairs_prob = jnp.array(pairs_prob)


    grids = jnp.array([
        positions_prob[6],
        positions_prob[6],
        positions_prob[6]
    ])

    ene = pot(positions, box, pairs, H.paramset.parameters)
    esp = es_pot(positions, grids, H.paramset.parameters["ADMPPmeForce"]["Q_local"])
    print("ESP:", esp)

    ene_prob = pot_prob(positions_prob, box, pairs_prob, H.paramset.parameters)
    print("Eprob:", ene_prob - ene)
    np.testing.assert_allclose(ene_prob - ene, esp[0], rtol=1e-3)