import openmm.app as app
import openmm.unit as unit
from typing import Tuple
import numpy as np
import jax.numpy as jnp
from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.xmlio import XMLIO
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException, isinstance_jnp
from ..admp.qeq import ADMPQeqForce
from ..generators.classical import CoulombGenerator
from ..admp.qeq import ADMPQeqForce
from ..admp.pme import setup_ewald_parameters


class ADMPQeqGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "ADMPQeqForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.coulomb14scale = float(
            self.ffinfo["Forces"][self.name]["meta"]["coulomb14scale"]
        )

        self.key_type = None
        keys, params = [], []
        qeq_mask = []
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

            if "mask" in node["attrib"] and node["attrib"]["mask"].upper() == "TRUE":
                qeq_mask.append(0.0)
            else:
                qeq_mask.append(1.0)

            params.append([chi0, J0, eta0])

        self.atom_keys = keys
        qeq_mask = jnp.array(qeq_mask)
        chi = jnp.array([i[0] for i in params])
        J = jnp.array([i[1] for i in params])
        eta = jnp.array([i[2] for i in params])

        paramset.addParameter(chi, "chi", field=self.name, mask=qeq_mask)
        paramset.addParameter(J, "J", field=self.name, mask=qeq_mask)
        paramset.addParameter(eta, "eta", field=self.name, mask=qeq_mask)
        # default params
        self._jaxPotential = None
        meta = self.ffinfo["Forces"][self.name]["meta"]
        if "DampMod" in meta:
            self.damp_mod = int(meta["DampMod"])
        else:
            self.damp_mod = 3

    def getName(self) -> str:
        """
        Returns the name of the force field.

        Returns:
        --------
        str
            The name of the force field.
        """
        return self.name

    def overwrite(self, paramset: ParamSet) -> None:
        node_indices = [
            i
            for i in range(len(self.ffinfo["Forces"][self.name]["node"]))
            if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Atom"
        ]
        chi = paramset[self.name]["chi"]
        J = paramset[self.name]["J"]
        eta = paramset[self.name]["eta"]
        atom_mask = paramset.mask[self.name]["chi"]
        for nidx, idx in enumerate(node_indices):
            chi0 = chi[nidx]
            J0 = J[nidx]
            eta0 = eta[nidx]
            mask = atom_mask[nidx]
            self.ffinfo["Forces"][self.name]["node"][idx]["attrib"]["chi"] = chi0
            self.ffinfo["Forces"][self.name]["node"][idx]["attrib"]["J"] = J0
            self.ffinfo["Forces"][self.name]["node"][idx]["attrib"]["eta"] = eta0
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][idx]["attrib"]["mask"] = "true"

    def _find_atype_key_index(self, atype: str):
        for n, i in enumerate(self.atom_keys):
            if i == atype:
                return n
        return None

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        methodMap = {
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise DMFFException("Illegal nonbonded method for NonbondedForce")

        # setting for coul force
        isNoCut = False
        if nonbondedMethod is app.NoCutoff:
            isNoCut = True

        mscales_coul = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # mscale for PME
        mscales_coul = mscales_coul.at[2].set(self.coulomb14scale)
        self.mscales_coul = mscales_coul  # for qeq calculation

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff

        if not isNoCut:
            cell = topdata.getPeriodicBoxVectors()
            box = jnp.array(cell)
            self.ethresh = kwargs.get("ethresh", 1e-5)
            self.coeff_method = kwargs.get("PmeCoeffMethod", "openmm")
            self.fourier_spacing = kwargs.get("PmeSpacing", 0.1)
            kappa, K1, K2, K3 = setup_ewald_parameters(
                r_cut, self.ethresh, box, self.fourier_spacing, self.coeff_method
            )
        else:
            kappa, K1, K2, K3 = 1.0, 1, 1, 1
        K = (K1, K2, K3)

        neutral_flag = kwargs.get("neutral", True)
        slab_flag = kwargs.get("slab", False)
        constQ = kwargs.get("constQ", True)
        part_const = kwargs.get("part_const", True)

        # top info
        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        residues = [r for r in topdata.residues()]
        n_residues = len(residues)
        init_q = np.array([a.meta["charge"] for a in atoms])
        map_idx = []
        for natom, atom in enumerate(atoms):
            atype = atom.meta[self.key_type]
            map_idx.append(self._find_atype_key_index(atype))
        map_idx = jnp.array(map_idx)

        if "const_list" in kwargs and "const_vals" in kwargs:
            const_list = kwargs["const_list"]
            const_vals = kwargs["const_vals"]
        else:
            const_list = []
            const_vals = []
            for r in residues:
                aidx = [a.index for a in r.atoms()]
                const_list.append(aidx)
                const_vals.append(sum(init_q[aidx]))

        has_aux = False
        if "has_aux" in kwargs:
            has_aux = kwargs["has_aux"]

        qeq_force = ADMPQeqForce(
            init_q,
            r_cut,
            kappa,
            K,
            damp_mod=self.damp_mod,
            const_list=const_list,
            const_vals=const_vals,
            neutral_flag=neutral_flag,
            slab_flag=slab_flag,
            constQ=constQ,
            pbc_flag=(not isNoCut),
            part_const=part_const,
            has_aux=has_aux,
        )
        qeq_energy = qeq_force.generate_get_energy()

        mscales_coul = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # mscale for PME
        mscales_coul = mscales_coul.at[2].set(self.coulomb14scale)

        def potential_fn(
            positions: jnp.ndarray,
            box: jnp.ndarray,
            pairs: jnp.ndarray,
            params: ParamSet,
            aux: dict = None
        ) -> jnp.ndarray:
            # map_atomtype = np.zeros(n_atoms)
            eta = params[self.name]["eta"][map_idx]
            chi = params[self.name]["chi"][map_idx]
            J = params[self.name]["J"][map_idx]
            if has_aux:
                qeq_energy0, aux = qeq_energy(positions, box, pairs, mscales_coul, eta, chi, J, aux)
                # return pme_energy + qeq_energy0
                return qeq_energy0, aux
            else:
                qeq_energy0 = qeq_energy(positions, box, pairs, mscales_coul, eta, chi, J )
                # return pme_energy + qeq_energy0
                return qeq_energy0

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["ADMPQeqForce"] = ADMPQeqGenerator
