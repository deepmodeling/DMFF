from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException, isinstance_jnp
from ..admp.pme import setup_ewald_parameters
import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from ..classical.inter import CoulNoCutoffForce, CoulombPMEForce, CoulReactionFieldForce, LennardJonesForce


class CoulombGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "CoulombForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.coulomb14scale = float(
            self.ffinfo["Forces"]["CoulombForce"]["meta"]["coulomb14scale"])
        self._use_bcc = False
        self._bcc_mol = []
        self.bcc_parsers = []
        bcc_prms = []
        bcc_mask = []
        for node in self.ffinfo["Forces"]["CoulombForce"]["node"]:
            if node["name"] == "UseBondChargeCorrection":
                self._use_bcc = True
                self._bcc_mol.append(node["attrib"]["name"])
            if node["name"] == "BondChargeCorrection":
                bcc = node["attrib"]["bcc"]
                parser = node["attrib"]["smarts"] if "smarts" in node["attrib"] else node["attrib"]["smirks"]
                bcc_prms.append(float(bcc))
                self.bcc_parsers.append(parser)
                if "mask" in node["attrib"] and node["attrib"]["mask"].upper() == "TRUE":
                    bcc_mask.append(0.0)
                else:
                    bcc_mask.append(1.0)
        bcc_prms = jnp.array(bcc_prms)
        bcc_mask = jnp.array(bcc_mask)
        paramset.addParameter(bcc_prms, "bcc", field=self.name, mask=bcc_mask)
        self._bcc_shape = paramset[self.name]["bcc"].shape[0]

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        # paramset to ffinfo
        if self._use_bcc:
            bcc_now = paramset[self.name]["bcc"]
            nbcc = 0
            for nnode, node in enumerate(self.ffinfo["Forces"][self.name]["node"]):
                if node["name"] == "BondChargeCorrection":
                    self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["bcc"] = bcc_now[nbcc]
                    nbcc += 1

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, args):
        methodMap = {
            app.NoCutoff: "NoCutoff",
            app.CutoffPeriodic: "CutoffPeriodic",
            app.CutoffNonPeriodic: "CutoffNonPeriodic",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise DMFFException("Illegal nonbonded method for NonbondedForce")

        isNoCut = False
        if nonbondedMethod is app.NoCutoff:
            isNoCut = True

        mscales_coul = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0,
                                  1.0])  # mscale for PME
        mscales_coul = mscales_coul.at[2].set(self.coulomb14scale)

        # set PBC
        if nonbondedMethod not in [app.NoCutoff, app.CutoffNonPeriodic]:
            ifPBC = True
        else:
            ifPBC = False

        charges = [a.meta["charge"] for a in topdata.atoms()]
        charges = jnp.array(charges)

        cov_mat = topdata.buildCovMat()

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff

        # PME Settings
        if nonbondedMethod is app.PME:
            cell = topdata.getPeriodicBoxVectors()
            box = jnp.array(cell)
            self.ethresh = args.get("ethresh", 1e-6)
            self.coeff_method = args.get("PmeCoeffMethod", "openmm")
            self.fourier_spacing = args.get("PmeSpacing", 0.1)
            kappa, K1, K2, K3 = setup_ewald_parameters(r_cut, self.ethresh,
                                                       box,
                                                       self.fourier_spacing,
                                                       self.coeff_method)

        if self._use_bcc:
            top_mat = np.zeros(
                (topdata.getNumAtoms(), self._bcc_shape))
            matched_dict = {}
            for nparser, parser in enumerate(self.bcc_parsers):
                matches = topdata.parseSMARTS(parser, resname=self._bcc_mol)
                for ii, jj in matches:
                    if (ii, jj) in matched_dict:
                        del matched_dict[(ii, jj)]
                    elif (jj, ii) in matched_dict:
                        del matched_dict[(jj, ii)]
                    matched_dict[(ii, jj)] = nparser
            for ii, jj in matched_dict.keys():
                nval = matched_dict[(ii, jj)]
                top_mat[ii, nval] += 1.
                top_mat[jj, nval] -= 1.

        topdata._meta["bcc_top_mat"] = top_mat

        if nonbondedMethod is not app.PME:
            # do not use PME
            if nonbondedMethod in [app.CutoffPeriodic, app.CutoffNonPeriodic]:
                # use Reaction Field
                coulforce = CoulReactionFieldForce(
                    r_cut,
                    isPBC=ifPBC,
                    topology_matrix=top_mat if self._use_bcc else None)
            if nonbondedMethod is app.NoCutoff:
                # use NoCutoff
                coulforce = CoulNoCutoffForce(
                    topology_matrix=top_mat if self._use_bcc else None)
        else:
            coulforce = CoulombPMEForce(
                r_cut,
                kappa, (K1, K2, K3),
                topology_matrix=top_mat if self._use_bcc else None)

        coulenergy = coulforce.generate_get_energy()

        def potential_fn(positions, box, pairs, params):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)

            if self._use_bcc:
                coulE = coulenergy(positions, box, pairs, charges,
                                   params["CoulombForce"]["bcc"], mscales_coul)
            else:
                coulE = coulenergy(positions, box, pairs, charges,
                                   mscales_coul)

            return coulE

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["CoulombForce"] = CoulombGenerator


class LennardJonesGenerator:

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "LennardJonesForce"
        self.ffinfo = ffinfo
        self.lj14scale = float(
            self.ffinfo["Forces"][self.name]["meta"]["lj14scale"])
        self.nbfix_to_idx = {}
        self.atype_to_idx = {}
        sig_prms, eps_prms = [], []
        sig_mask, eps_mask = [], []
        sig_nbfix, eps_nbfix = [], []
        sig_nbf_mask, eps_nbf_mask = [], []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                if "type" in node["attrib"]:
                    atype, eps, sig = node["attrib"]["type"], node["attrib"][
                        "epsilon"], node["attrib"]["sigma"]
                    self.atype_to_idx[atype] = len(sig_prms)
                elif "class" in node["attrib"]:
                    acls, eps, sig = node["attrib"]["class"], node["attrib"][
                        "epsilon"], node["attrib"]["sigma"]
                    atypes = ffinfo["ClassToType"][acls]
                    for atype in atypes:
                        self.atype_to_idx[atype] = len(sig_prms)
                sig_prms.append(float(sig))
                eps_prms.append(float(eps))
                if "mask" in node["attrib"] and node["attrib"]["mask"].upper() == "TRUE":
                    sig_mask.append(0.0)
                    eps_mask.append(0.0)
                else:
                    sig_mask.append(1.0)
                    eps_mask.append(1.0)
            elif node["name"] == "NBFixPair":
                if "type1" in node["attrib"]:
                    atype1, atype2, eps, sig = node["attrib"]["type1"], node["attrib"][
                        "type2"], node["attrib"]["epsilon"], node["attrib"]["sigma"]
                    if atype1 not in self.nbfix_to_idx:
                        self.nbfix_to_idx[atype1] = {}
                    if atype2 not in self.nbfix_to_idx:
                        self.nbfix_to_idx[atype2] = {}
                    self.nbfix_to_idx[atype1][atype2] = len(sig_nbfix)
                    self.nbfix_to_idx[atype2][atype1] = len(sig_nbfix)
                elif "class1" in node["attrib"]:
                    acls1, acls2, eps, sig = node["attrib"]["class1"], node["attrib"][
                        "class2"], node["attrib"]["epsilon"], node["attrib"]["sigma"]
                    atypes1 = ffinfo["ClassToType"][acls1]
                    atypes2 = ffinfo["ClassToType"][acls2]
                    for atype1 in atypes1:
                        if atype1 not in self.nbfix_to_idx:
                            self.nbfix_to_idx[atype1] = {}
                        for atype2 in atypes2:
                            if atype2 not in self.nbfix_to_idx:
                                self.nbfix_to_idx[atype2] = {}
                            self.nbfix_to_idx[atype1][atype2] = len(sig_nbfix)
                            self.nbfix_to_idx[atype2][atype1] = len(sig_nbfix)
                sig_nbfix.append(float(sig))
                eps_nbfix.append(float(eps))
                if "mask" in node["attrib"] and node["attrib"]["mask"].upper() == "TRUE":
                    sig_nbf_mask.append(0.0)
                    eps_nbf_mask.append(0.0)
                else:
                    sig_nbf_mask.append(1.0)
                    eps_nbf_mask.append(1.0)

        sig_prms = jnp.array(sig_prms)
        eps_prms = jnp.array(eps_prms)
        sig_mask = jnp.array(sig_mask)
        eps_mask = jnp.array(eps_mask)

        sig_nbfix, eps_nbfix = jnp.array(sig_nbfix), jnp.array(eps_nbfix)
        sig_nbf_mask = jnp.array(sig_nbf_mask)
        eps_nbf_mask = jnp.array(eps_nbf_mask)

        paramset.addField(self.name)
        paramset.addParameter(sig_prms, "sigma", field=self.name, mask=sig_mask)
        paramset.addParameter(eps_prms, "epsilon", field=self.name, mask=eps_mask)
        paramset.addParameter(sig_nbfix, "sigma_nbfix", field=self.name, mask=sig_nbf_mask)
        paramset.addParameter(eps_nbfix, "epsilon_nbfix", field=self.name, mask=eps_nbf_mask)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        # paramset to ffinfo
        for nnode in range(len(self.ffinfo["Forces"][self.name]["node"])):
            node = self.ffinfo["Forces"][self.name]["node"][nnode]
            if node["name"] == "Atom":
                if "type" in node["attrib"]:
                    atype = node["attrib"]["type"]
                    idx = self.atype_to_idx[atype]

                elif "class" in node["attrib"]:
                    acls = node["attrib"]["class"]
                    atypes = self.ffinfo["ClassToType"][acls]
                    idx = self.atype_to_idx[atypes[0]]

                eps_now = paramset[self.name]["epsilon"][idx]
                sig_now = paramset[self.name]["sigma"][idx]
                self.ffinfo["Forces"][
                    self.name]["node"][nnode]["attrib"]["sigma"] = sig_now
                self.ffinfo["Forces"][
                    self.name]["node"][nnode]["attrib"]["epsilon"] = eps_now
            # have not tested for NBFixPair overwrite
            elif node["name"] == "NBFixPair":
                if "type1" in node["attrib"]:
                    atype1, atype2 = node["attrib"]["type1"], node["attrib"]["type2"]
                    idx = self.nbfix_to_idx[atype1][atype2]
                elif "class1" in node["attrib"]:
                    acls1, acls2 = node["attrib"]["class1"], node["attrib"]["class2"]
                    atypes1 = self.ffinfo["ClassToType"][acls1]
                    atypes2 = self.ffinfo["ClassToType"][acls2]
                    idx = self.nbfix_to_idx[atypes1[0]][atypes2[0]]
                sig_now = paramset[self.name]["sigma_nbfix"][idx]
                eps_now = paramset[self.name]["epsilon_nbfix"][idx]
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["sigma"] = sig_now
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["epsilon"] = eps_now

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, args):
        methodMap = {
            app.NoCutoff: "NoCutoff",
            app.CutoffPeriodic: "CutoffPeriodic",
            app.CutoffNonPeriodic: "CutoffNonPeriodic",
            app.PME: "CutoffPeriodic",
        }
        if nonbondedMethod not in methodMap:
            raise DMFFException("Illegal nonbonded method for NonbondedForce")
        methodString = methodMap[nonbondedMethod]

        atoms = [a for a in topdata.atoms()]
        atypes = [a.meta["type"] for a in atoms]
        map_prm = []
        for atype in atypes:
            if atype not in self.atype_to_idx:
                raise DMFFException(f"Atom type {atype} not found.")
            idx = self.atype_to_idx[atype]
            map_prm.append(idx)
        map_prm = jnp.array(map_prm)
        topdata._meta["lj_map_idx"] = map_prm

        # not use nbfix for now
        map_nbfix = []
        for atype1 in self.nbfix_to_idx.keys():
            for atype2 in self.nbfix_to_idx[atype1].keys():
                nbfix_idx = self.nbfix_to_idx[atype1][atype2]
                type1_idx = self.atype_to_idx[atype1]
                type2_idx = self.atype_to_idx[atype2]
                map_nbfix.append([type1_idx, type2_idx, nbfix_idx])
        map_nbfix = np.array(map_nbfix, dtype=int).reshape((-1, 3))

        if methodString in ["NoCutoff", "CutoffNonPeriodic"]:
            isPBC = False
            if methodString == "NoCutoff":
                isNoCut = True
            else:
                isNoCut = False
        else:
            isPBC = True
            isNoCut = False

        mscales_lj = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # mscale for LJ
        mscales_lj = mscales_lj.at[2].set(self.lj14scale)

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff

        ljforce = LennardJonesForce(0.0,
                                    r_cut,
                                    map_prm,
                                    map_nbfix,
                                    isSwitch=False,
                                    isPBC=isPBC,
                                    isNoCut=isNoCut)
        ljenergy = ljforce.generate_get_energy()

        def potential_fn(positions, box, pairs, params):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)

            ljE = ljenergy(positions, box, pairs,
                           params[self.name]["epsilon"],
                           params[self.name]["sigma"],
                           params[self.name]["epsilon_nbfix"],
                           params[self.name]["sigma_nbfix"],
                           mscales_lj)

            return ljE

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["LennardJonesForce"] = LennardJonesGenerator
