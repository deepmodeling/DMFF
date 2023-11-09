#!/usr/bin/env python
# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) 
# and the index of each atom. All the information are required to store in the tensor of torch. 
# Then, you just pass these information to the calss "pes" that will output the energy and force.
import numpy as np
import torch
import sys
import MDAnalysis as mda
from openmm import *
from openmm.app import *

if __name__ == '__main__':
    # used for select a unoccupied GPU
    # gpu/cpu
    device = torch.device('cpu')

    # same as the atomtype in the file input_density
    atomtype=['H', 'C', 'O']

    # set up force calculators
    mol = PDBFile('peg4.pdb')
    pos = np.array(mol.positions._value) * 10
    box = np.array(mol.topology.getPeriodicBoxVectors()._value) * 10
    species = []
    mass = []
    # Loop over all atoms in the topology
    for atom in mol.topology.atoms():
        # Get the element of the atom
        element = atom.element.symbol
        mass.append(atom.element.mass._value)
        species.append(atomtype.index(atom.element.symbol))

    #load the serilizable model
    pes=torch.jit.load('EANN_PES_DOUBLE.pt')
    # FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
    pes.to(device).to(torch.double)
    # set the eval mode
    pes.eval()
    pes=torch.jit.optimize_for_inference(pes)
    # save the lattic parameters
    period_table=torch.tensor([1,1,1],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
    species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
    cart=torch.from_numpy(np.array(pos)).to(device).to(torch.double)  # also float32/double
    mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
    tcell=torch.from_numpy(box).to(device).to(torch.double)  # also float32/double
    energy,force=pes(period_table,cart,tcell,species,mass)
    energy=energy.detach()
    force=force.detach()
    print('# Reference Energy:', float(energy))
