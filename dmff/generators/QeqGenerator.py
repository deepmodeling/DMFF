#!/usr/bin/env python

import openmm.app as app
import openmm.unit as unit
from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax
from dmff.api.topology import DMFFTopology
from dmff.api.paramset import ParamSet
from dmff.api.xmlio import XMLIO
from dmff.api.hamiltonian import _DMFFGenerators
from dmff.utils import DMFFException, isinstance_jnp
from dmff.admp.qeq import ADMPQeqForce 
from dmff.generators.classical import CoulombGenerator
from dmff.admp import qeq 


class ADMPQeqGenerator:
    def __init__(self, ffinfo:dict, paramset: ParamSet):

        self.name = 'ADMPQeqForce'
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None
        keys , params = [], []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            attribs = node["attrib"]
            
            if self.key_type is None and "type" in attribs:
                self.key_type = "type"
            elif self.key_type is None and "class" in attribs:
                self.key_type = "class"
            elif self.key_type is not None and f"{self.key_type}" not in attribs:
                raise ValueError("Keyword 'class' or 'type' cannot be used together.")
            elif self.key_type is not None and f"{self.key_type}" in attribs:
                pass
            else:
                raise ValueError("Cannot find key type for ADMPQeqForce.")
            key = attribs[self.key_type]
            keys.append(key)

            chi0 = float(attribs["chi"]) 
            J0 = float(attribs["J"])
            eta0 = float(attribs["eta"])

            params.append([chi0, J0, eta0])

        self.keys = keys
        chi = jnp.array([i[0] for i in params])
        J = jnp.array([i[1] for i in params])
        eta = jnp.array([i[2] for i in params])

        paramset.addParameter(chi, "chi", field=self.name)
        paramset.addParameter(J, "J", field=self.name)
        paramset.addParameter(eta, "eta", field=self.name)
        # default params
        self._jaxPotential = None
        self.damp_mod = self.ffinfo["Forces"][self.name]["meta"]["DampMod"] 
        self.neutral_flag = self.ffinfo["Forces"][self.name]["meta"]["NeutralFlag"]
        self.slab_flag = self.ffinfo["Forces"][self.name]["meta"]["SlabFlag"]
        self.constQ = self.ffinfo["Forces"][self.name]["meta"]["ConstQFlag"]
        self.pbc_flag = self.ffinfo["Forces"][self.name]["meta"]["PbcFlag"] 

        self.pme_generator = CoulombGenerator(ffinfo, paramset)

    def getName(self) -> str:
        """
        Returns the name of the force field.

        Returns:
        --------
        str
            The name of the force field.
        """
        return self.name
    
    def overwrite(self, paramset:ParamSet) -> None:

        node_indices = [ i for i in range(len(self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "QeqAtom"]
        chi = paramset[self.name]["chi"]
        J = paramset[self.name]["J"]
        eta = paramset[self.name]["eta"]
        for nnode, key in enumerate(self.keys):
            self.ffinfo["Forces"][self.name]["node"][node_indices[nnode]]["attrib"] = {}
            self.ffinfo["Forces"][self.name]["node"][node_indices[nnode]]["attrib"][f"{self.key_type}"] = key
            chi0 = chi[nnode]
            J0 = J[nnode]
            eta0 = eta[nnode]
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]]["attrib"]["chi"] = str(chi0)
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]]["attrib"]["J"] = str(J0)
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]]["attrib"]["eta"] = str(eta0)


    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, charges, const_list, const_vals, map_atomtype):

        n_atoms = topdata._numAtoms
        n_residues = topdata._numResidues
        
        q = jnp.array(charges)
        lagmt = np.ones(n_residues)
        b_value = jnp.concatenate((q,lagmt))
        qeq_force = ADMPQeqForce(q, lagmt,self.damp_mod, self.neutral_flag,
                                 self.slab_flag, self.constQ, self.pbc_flag)
        self.qeq_force = qeq_force
        qeq_energy = qeq_force.generate_get_energy()

        self.pme_potential = self.pme_generator.createPotential(topdata, app.PME, nonbondedCutoff )  
        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet) -> jnp.ndarray:
            
            n_atoms = len(positions)
           # map_atomtype = np.zeros(n_atoms)
            eta = np.array(params[self.name]["eta"])[map_atomtype]
            chi = np.array(params[self.name]["chi"])[map_atomtype]
            J = np.array(params[self.name]["J"])[map_atomtype]
            self.eta = jnp.array(eta)
            self.chi = jnp.array(chi)
            self.J = jnp.array(J)
         #   coulenergy = self.pme_generator.coulenergy
         #   pme_energy = pme_potential(positions, box, pairs, params)
            damp_mod = self.damp_mod
            neutral_flag = self.neutral_flag
            constQ = self.constQ
            pbc_flag = self.pbc_flag
            
            qeq_energy0 = qeq_energy(positions, box, pairs, q, lagmt,
                                    eta, chi, J,const_list, 
                                    const_vals, self.pme_generator) 
           # return pme_energy + qeq_energy0
            return qeq_energy0

        self._jaxPotential = potential_fn
        return potential_fn

_DMFFGenerators["ADMPQeqForce"] = ADMPQeqGenerator
