#!/usr/bin/env python
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
import numpy as np
import jax
import jax.numpy as jnp
import torch
from torch2jax import j2t, t2j
from dmff.torch_tools import wrap_torch_potential_kernel, j2t_pytree, t2j_pytree


# manual implementation of the LJ kernel
atomtypes = np.array((0, 0, 1))
def potential_t(positions, box, pairs, params):
    sigmas_atoms = params["NonbondedForce"]["sigma"][atomtypes]
    epsilons_atoms = params["NonbondedForce"]["epsilon"][atomtypes]
    sigma1_list = sigmas_atoms[pairs[:, 0]]
    sigma2_list = sigmas_atoms[pairs[:, 1]]
    epsilon1_list = epsilons_atoms[pairs[:, 0]]
    epsilon2_list = epsilons_atoms[pairs[:, 1]]
    # sigma_list = torch.sqrt(sigma1_list * sigma2_list)
    sigma_list = (sigma1_list + sigma2_list)/2
    epsilon_list = torch.sqrt(epsilon1_list * epsilon2_list)
    r1 = positions[pairs[:, 0]]
    r2 = positions[pairs[:, 1]]
    dr = r1 - r2
    dr2 = torch.sum(dr * dr, dim=1)
    sigma_dr2 = sigma_list * sigma_list / dr2
    energies = 4 * epsilon_list * ((sigma_dr2)**6 - (sigma_dr2)**3)
    return torch.sum(energies)


if __name__ == '__main__':
    H = Hamiltonian('params.xml')
    params = H.getParameters().parameters
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("structure.pdb")
    rc = 1.0

    pots = H.createPotential(pdb.topology, \
                             nonbondedMethod=app.CutoffPeriodic, \
                             nonbondedCutoff=rc*unit.nanometer)

    potential_jax = pots.dmff_potentials['NonbondedForce']

    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map']) 
    nbl.allocate(positions)


    # native jax implementation
    pairs = np.array(nbl.pairs)
    ene, p_grad = jax.value_and_grad(potential_jax, argnums=3)(positions, box, nbl.pairs, params)

    print("JAX kernel results:")
    print(p_grad["NonbondedForce"]["sigma"])
    print(p_grad["NonbondedForce"]["epsilon"])

    # torch input and kernel
    positions_t = j2t_pytree(positions)
    box_t = j2t_pytree(box)
    params_t = j2t_pytree(params)
    ene = potential_t(positions_t, box_t, pairs, params_t)
    ene.backward()
    print("Torch kernel results:")
    print(params_t["NonbondedForce"]["sigma"].grad)
    print(params_t["NonbondedForce"]["epsilon"].grad)

    # Wrapping torch with JAX
    potential_wrapped = wrap_torch_potential_kernel(potential_t)
    ene, p_grad = jax.value_and_grad(potential_wrapped, argnums=3)(positions, box, nbl.pairs, params)
    print("Wrapped Torch kernel results:")
    print(p_grad["NonbondedForce"]["sigma"])
    print(p_grad["NonbondedForce"]["epsilon"])
