#!/usr/bin/env python
import sys
from pathlib import Path
admp_path = Path(__file__).parent.parent.parent.parent
print(admp_path)
sys.path.append(str(admp_path))
import openmm.app as app
import openmm.unit as unit
import jax.numpy as jnp
from dmff.api import Hamiltonian
from jax_md import space, partition
import jax
from dmff import settings

print(f'if DO_JIT: {settings.DO_JIT}')

pdb = 'water1024.pdb'
xml = 'forcefield.xml'

H = Hamiltonian(xml)
pdb = app.PDBFile(pdb)

rc = 4.0  # in Angstrom

generator = H.getGenerators()
disp_generator = generator[0]
pme_generator = generator[1]    

pme_generator.lpol = True # debug
pme_generator.ref_dip = 'disp_1024'
potential = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)

disp_pot, pme_pot = potential

positions = jnp.array(pdb.positions._value) * 10
a, b, c = pdb.topology.getPeriodicBoxVectors()
box = jnp.array([a._value, b._value, c._value]) * 10

# we omit data-parse stage because it is only execute once at the first of of program and not the bottleneck of the program.

displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
nbr = neighbor_list_fn.allocate(positions)
pairs = nbr.idx.T

n_atoms = len(positions)

pme_pot(positions, box, pairs, pme_generator.params)
disp_pot(positions, box, pairs, disp_generator.params)

jax.profiler.start_trace('./profile_log')

# electrostatic
pme_pot(positions, box, pairs, pme_generator.params)

# dispersion
disp_pot(positions, box, pairs, disp_generator.params)

# induced


jax.profiler.stop_trace()
print('done')