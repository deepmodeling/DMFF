#!/usr/bin/env python
import sys
import numpy as np
import torch
import torch.nn as nn
import MDAnalysis as mda
import copy


class ForceModel(torch.nn.Module):
    
    def __init__(self):
        super(ForceModel, self).__init__()
        # The parameter will be named 'my_bias'
        torch.manual_seed(12345)
        self.k = nn.Parameter((torch.randn(3)*0.3 + 1.0) * 10)
        self.r0 = nn.Parameter((torch.rand(3)) * 0.01 + 0.3) # in nm

    def forward(self, positions, box, atomtypes):
        # pairwise parameters
        ks = torch.zeros(3)
        r0s = torch.zeros(3)
        ks[0:2] = self.k[0]
        ks[2] = self.k[1]
        r0s[0:2] = self.r0[0]
        r0s[2] = self.r0[1]
        # pairwise distances
        rij = torch.zeros(3, 3)
        rij[0] = positions[0] - positions[1]
        rij[1] = positions[0] - positions[2]
        rij[2] = positions[1] - positions[2]
        rijnorm = torch.norm(rij, dim=1)
        rij_normed = (rij.T / rijnorm).T
        energy = torch.sum(0.5 * ks * (rijnorm - r0s)**2)
        fij = torch.zeros(3, 3)
        fij = (-ks * (rijnorm - r0s) * rij_normed.T).T
        forces = torch.zeros(3, 3)
        forces[0] = fij[0] + fij[1]
        forces[1] = -fij[0] + fij[2]
        forces[2] = -fij[1] - fij[2]
        virial = rij.T @ fij
        results = {}
        results['pred_energy'] = energy
        results['pred_forces'] = forces
        results['pred_virial'] = virial
        return results


if __name__ == "__main__":
    model = ForceModel()
    u = mda.Universe('structure.pdb')
    positions = torch.tensor(u.atoms.positions / 10)
    box = torch.eye(3) * 3.0
    atomtypes = np.array([0, 0, 1])
    res = model(positions, box, atomtypes)
    print('Predicted Energy')
    print(res['pred_energy'])
    res['pred_energy'].backward()
    print('Parameter Gradient')
    for pname, p in model.named_parameters():
        print(pname, p.grad)
    print('Forces:')
    print(res['pred_forces'])
    # # print(res['pred_forces'])
    # print(res['pred_virial'])
    # # check force
    # x0 = copy.deepcopy(positions[1, 1])
    # print('#', res['pred_forces'][1, 1])
    # delta = 0.0005
    # energies = torch.zeros(5)
    # for i in range(-2, 3):
    #     positions[1, 1] = x0 + delta * i
    #     res = model(positions, box, atomtypes)
    #     energies[i+2] = res['pred_energy']
    #     # print(i*delta, '%.8f'%res['pred_energy'])
    # print((energies[-1] - energies[0])/delta/4)
    # # check virial

    # save model
    torch.save(model, 'model.pt')
