from collections import defaultdict
from typing import Dict

import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit

import dmff
from dmff.classical.intra import (
    HarmonicBondJaxForce,
    HarmonicAngleJaxForce,
    PeriodicTorsionJaxForce,
)
from dmff.classical.inter import (
    LennardJonesForce,
    LennardJonesLongRangeForce,
    CoulombPMEForce,
    CoulNoCutoffForce,
    CoulombPMEForce,
    CoulReactionFieldForce,
    LennardJonesForce,
)
from dmff.classical.fep import (
    LennardJonesFreeEnergyForce,
    LennardJonesLongRangeFreeEnergyForce,
    CoulombPMEFreeEnergyForce
)
from dmff.admp.pme import  setup_ewald_parameters
from dmff.utils import jit_condition, isinstance_jnp, DMFFException, findItemInList
from dmff.fftree import ForcefieldTree, TypeMatcher
from dmff.api import Hamiltonian, build_covalent_map


class HarmonicBondJaxGenerator:
    def __init__(self, ff: Hamiltonian):
        self.name = "HarmonicBondForce"
        self.ff: Hamiltonian = ff
        self.fftree: ForcefieldTree = ff.fftree
        self.paramtree: Dict = ff.paramtree

    def extract(self):
        """
        extract forcefield paramters from ForcefieldTree. 
        """
        lengths = self.fftree.get_attribs(f"{self.name}/Bond", "length")
        # get_attribs will return a list of list.
        ks = self.fftree.get_attribs(f"{self.name}/Bond", "k")
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["length"] = jnp.array(lengths)
        self.paramtree[self.name]["k"] = jnp.array(ks)

    def overwrite(self):
        """
        update parameters in the fftree by using paramtree of this generator.
        """
        self.fftree.set_attrib(f"{self.name}/Bond", "length",
                               self.paramtree[self.name]["length"])
        self.fftree.set_attrib(f"{self.name}/Bond", "k",
                               self.paramtree[self.name]["k"])

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        This method will create a potential calculation kernel. It usually should do the following:
        
        1. Match the corresponding bond parameters according to the atomic types at both ends of each bond.

        2. Create a potential calculation kernel, and pass those mapped parameters to the kernel.

        3. assign the jax potential to the _jaxPotential.

        Args:
            Those args are the same as those in createSystem.
        """

        # initialize typemap
        matcher = TypeMatcher(self.fftree, "HarmonicBondForce/Bond")

        map_atom1, map_atom2, map_param = [], [], []
        n_bonds = len(data.bonds)
        # build map
        for i in range(n_bonds):
            idx1 = data.bonds[i].atom1
            idx2 = data.bonds[i].atom2
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            ifFound, ifForward, nfunc = matcher.matchGeneral([type1, type2])
            if not ifFound:
                raise BaseException(
                    f"No parameter for bond ({idx1},{type1}) - ({idx2},{type2})"
                )
            map_atom1.append(idx1)
            map_atom2.append(idx2)
            map_param.append(nfunc)
        map_atom1 = np.array(map_atom1, dtype=int)
        map_atom2 = np.array(map_atom2, dtype=int)
        map_param = np.array(map_param, dtype=int)

        bforce = HarmonicBondJaxForce(map_atom1, map_atom2, map_param)

        def potential_fn(positions, box, pairs, params):
            return bforce.get_energy(positions, box, pairs,
                                     params[self.name]["k"],
                                     params[self.name]["length"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


dmff.api.jaxGenerators["HarmonicBondForce"] = HarmonicBondJaxGenerator


class HarmonicAngleJaxGenerator:
    def __init__(self, ff):
        self.name = "HarmonicAngleForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree

    def extract(self):
        angles = self.fftree.get_attribs(f"{self.name}/Angle", "angle")
        ks = self.fftree.get_attribs(f"{self.name}/Angle", "k")
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["angle"] = jnp.array(angles)
        self.paramtree[self.name]["k"] = jnp.array(ks)

    def overwrite(self):
        self.fftree.set_attrib(f"{self.name}/Angle", "angle",
                               self.paramtree[self.name]["angle"])
        self.fftree.set_attrib(f"{self.name}/Angle", "k",
                               self.paramtree[self.name]["k"])

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        matcher = TypeMatcher(self.fftree, "HarmonicAngleForce/Angle")

        map_atom1, map_atom2, map_atom3, map_param = [], [], [], []
        n_angles = len(data.angles)
        for nangle in range(n_angles):
            idx1 = data.angles[nangle][0]
            idx2 = data.angles[nangle][1]
            idx3 = data.angles[nangle][2]
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            type3 = data.atomType[data.atoms[idx3]]
            ifFound, ifForward, nfunc = matcher.matchGeneral(
                [type1, type2, type3])
            if not ifFound:
                print(
                    f"No parameter for angle ({idx1},{type1}) - ({idx2},{type2}) - ({idx3},{type3})"
                )
            else:
                map_atom1.append(idx1)
                map_atom2.append(idx2)
                map_atom3.append(idx3)
                map_param.append(nfunc)
        map_atom1 = np.array(map_atom1, dtype=int)
        map_atom2 = np.array(map_atom2, dtype=int)
        map_atom3 = np.array(map_atom3, dtype=int)
        map_param = np.array(map_param, dtype=int)

        aforce = HarmonicAngleJaxForce(map_atom1, map_atom2, map_atom3,
                                       map_param)

        def potential_fn(positions, box, pairs, params):
            return aforce.get_energy(positions, box, pairs,
                                     params[self.name]["k"],
                                     params[self.name]["angle"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


dmff.api.jaxGenerators["HarmonicAngleForce"] = HarmonicAngleJaxGenerator


class PeriodicTorsionJaxGenerator:
    def __init__(self, ff):
        self.name = "PeriodicTorsionForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self.meta = {}

        self.meta["prop_order"] = defaultdict(list)
        self.meta["prop_nodeidx"] = defaultdict(list)

        self.meta["impr_order"] = defaultdict(list)
        self.meta["impr_nodeidx"] = defaultdict(list)

        self.max_pred_prop = 0
        self.max_pred_impr = 0

    def extract(self):
        propers = self.fftree.get_nodes("PeriodicTorsionForce/Proper")
        impropers = self.fftree.get_nodes("PeriodicTorsionForce/Improper")
        self.paramtree[self.name] = {}
        # propers
        prop_phase = defaultdict(list)
        prop_k = defaultdict(list)
        for nnode, node in enumerate(propers):
            for key in node.attrs:
                if "periodicity" in key:
                    order = int(key[-1])
                    phase = float(node.attrs[f"phase{order}"])
                    k = float(node.attrs[f"k{order}"])
                    periodicity = int(node.attrs[f"periodicity{order}"])
                    if self.max_pred_prop < periodicity:
                        self.max_pred_prop = periodicity
                    prop_phase[f"{periodicity}"].append(phase)
                    prop_k[f"{periodicity}"].append(k)
                    self.meta[f"prop_order"][f"{periodicity}"].append(order)
                    self.meta[f"prop_nodeidx"][f"{periodicity}"].append(nnode)

        self.paramtree[self.name]["prop_phase"] = {}
        self.paramtree[self.name]["prop_k"] = {}
        for npred in range(1, self.max_pred_prop + 1):
            self.paramtree[self.name]["prop_phase"][f"{npred}"] = jnp.array(
                prop_phase[f"{npred}"])
            self.paramtree[self.name]["prop_k"][f"{npred}"] = jnp.array(
                prop_k[f"{npred}"])
        if self.max_pred_prop == 0:
            del self.paramtree[self.name]["prop_phase"]
            del self.paramtree[self.name]["prop_k"]

        # impropers
        impr_phase = defaultdict(list)
        impr_k = defaultdict(list)
        for nnode, node in enumerate(impropers):
            for key in node.attrs:
                if "periodicity" in key:
                    order = int(key[-1])
                    phase = float(node.attrs[f"phase{order}"])
                    k = float(node.attrs[f"k{order}"])
                    periodicity = int(node.attrs[f"periodicity{order}"])
                    if self.max_pred_impr < periodicity:
                        self.max_pred_impr = periodicity
                    impr_phase[f"{periodicity}"].append(phase)
                    impr_k[f"{periodicity}"].append(k)
                    self.meta[f"impr_order"][f"{periodicity}"].append(order)
                    self.meta[f"impr_nodeidx"][f"{periodicity}"].append(nnode)

        self.paramtree[self.name]["impr_phase"] = {}
        self.paramtree[self.name]["impr_k"] = {}
        for npred in range(1, self.max_pred_impr + 1):
            self.paramtree[self.name]["impr_phase"][f"{npred}"] = jnp.array(
                impr_phase[f"{npred}"])
            self.paramtree[self.name]["impr_k"][f"{npred}"] = jnp.array(
                impr_k[f"{npred}"])
        if self.max_pred_impr == 0:
            del self.paramtree[self.name]["impr_phase"]
            del self.paramtree[self.name]["impr_k"]

    def overwrite(self):
        propers = self.fftree.get_nodes("PeriodicTorsionForce/Proper")
        impropers = self.fftree.get_nodes("PeriodicTorsionForce/Improper")
        prop_data = [{} for _ in propers]
        impr_data = [{} for _ in impropers]
        # make propers
        for periodicity in range(1, self.max_pred_prop + 1):
            nterms = len(
                self.paramtree[self.name][f"prop_phase"][f"{periodicity}"])
            for nitem in range(nterms):
                phase = self.paramtree[
                    self.name][f"prop_phase"][f"{periodicity}"][nitem]
                k = self.paramtree[
                    self.name][f"prop_k"][f"{periodicity}"][nitem]
                nodeidx = self.meta[f"prop_nodeidx"][f"{periodicity}"][nitem]
                order = self.meta[f"prop_order"][f"{periodicity}"][nitem]
                prop_data[nodeidx][f"phase{order}"] = phase
                prop_data[nodeidx][f"k{order}"] = k
        if "prop_phase" in self.paramtree[self.name]:
            self.fftree.set_node("PeriodicTorsionForce/Proper", prop_data)

        # make impropers
        for periodicity in range(1, self.max_pred_impr + 1):
            nterms = len(
                self.paramtree[self.name][f"impr_phase"][f"{periodicity}"])
            for nitem in range(nterms):
                phase = self.paramtree[
                    self.name][f"impr_phase"][f"{periodicity}"][nitem]
                k = self.paramtree[
                    self.name][f"impr_k"][f"{periodicity}"][nitem]
                nodeidx = self.meta[f"impr_nodeidx"][f"{periodicity}"][nitem]
                order = self.meta[f"impr_order"][f"{periodicity}"][nitem]
                impr_data[nodeidx][f"phase{order}"] = phase
                impr_data[nodeidx][f"k{order}"] = k
        if "impr_phase" in self.paramtree[self.name]:
            self.fftree.set_node("PeriodicTorsionForce/Improper", impr_data)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        proper_matcher = TypeMatcher(self.fftree,
                                     "PeriodicTorsionForce/Proper")
        map_prop_atom1 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_atom2 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_atom3 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_atom4 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_param = {i: [] for i in range(1, self.max_pred_prop + 1)}
        n_matched_props = 0
        for torsion in data.propers:
            types = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
            ifFound, ifForward, nnode = proper_matcher.matchGeneral(types)
            if not ifFound:
                continue
            # find terms for node
            for periodicity in range(1, self.max_pred_prop + 1):
                idx = findItemInList(
                    nnode, self.meta[f"prop_nodeidx"][f"{periodicity}"])
                if idx < 0:
                    continue
                n_matched_props += 1
                map_prop_atom1[periodicity].append(torsion[0])
                map_prop_atom2[periodicity].append(torsion[1])
                map_prop_atom3[periodicity].append(torsion[2])
                map_prop_atom4[periodicity].append(torsion[3])
                map_prop_param[periodicity].append(idx)

        impr_matcher = TypeMatcher(self.fftree,
                                   "PeriodicTorsionForce/Improper")
        try:
            ordering = self.fftree.get_attribs("PeriodicTorsionForce",
                                               "ordering")[0]
        except KeyError as e:
            ordering = "default"
        map_impr_atom1 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_atom2 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_atom3 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_atom4 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_param = {i: [] for i in range(1, self.max_pred_impr + 1)}
        n_matched_imprs = 0
        for impr in data.impropers:
            match = impr_matcher.matchImproper(impr, data, ordering=ordering)
            if match is not None:
                (a1, a2, a3, a4, nnode) = match
                n_matched_imprs += 1
                # find terms for node
                for periodicity in range(1, self.max_pred_impr + 1):
                    idx = findItemInList(
                        nnode, self.meta[f"impr_nodeidx"][f"{periodicity}"])
                    if idx < 0:
                        continue
                    if ordering == 'smirnoff':
                        # Add all torsions in trefoil
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a2)
                        map_impr_atom3[periodicity].append(a3)
                        map_impr_atom4[periodicity].append(a4)
                        map_impr_param[periodicity].append(idx)
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a3)
                        map_impr_atom3[periodicity].append(a4)
                        map_impr_atom4[periodicity].append(a2)
                        map_impr_param[periodicity].append(idx)
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a4)
                        map_impr_atom3[periodicity].append(a2)
                        map_impr_atom4[periodicity].append(a3)
                        map_impr_param[periodicity].append(idx)
                    else:
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a2)
                        map_impr_atom3[periodicity].append(a3)
                        map_impr_atom4[periodicity].append(a4)
                        map_impr_param[periodicity].append(idx)

        props = [
            PeriodicTorsionJaxForce(jnp.array(map_prop_atom1[p], dtype=int),
                                    jnp.array(map_prop_atom2[p], dtype=int),
                                    jnp.array(map_prop_atom3[p], dtype=int),
                                    jnp.array(map_prop_atom4[p], dtype=int),
                                    jnp.array(map_prop_param[p], dtype=int), p)
            for p in range(1, self.max_pred_prop + 1)
        ]
        imprs = [
            PeriodicTorsionJaxForce(jnp.array(map_impr_atom1[p], dtype=int),
                                    jnp.array(map_impr_atom2[p], dtype=int),
                                    jnp.array(map_impr_atom3[p], dtype=int),
                                    jnp.array(map_impr_atom4[p], dtype=int),
                                    jnp.array(map_impr_param[p], dtype=int), p)
            for p in range(1, self.max_pred_impr + 1)
        ]

        def potential_fn(positions, box, pairs, params):
            prop_sum = sum([
                props[i].get_energy(
                    positions, box, pairs,
                    params["PeriodicTorsionForce"]["prop_k"][f"{i+1}"],
                    params["PeriodicTorsionForce"]["prop_phase"][f"{i+1}"])
                for i in range(self.max_pred_prop)
            ])
            impr_sum = sum([
                imprs[i].get_energy(
                    positions, box, pairs,
                    params["PeriodicTorsionForce"]["impr_k"][f"{i+1}"],
                    params["PeriodicTorsionForce"]["impr_phase"][f"{i+1}"])
                for i in range(self.max_pred_impr)
            ])

            return prop_sum + impr_sum

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


dmff.api.jaxGenerators["PeriodicTorsionForce"] = PeriodicTorsionJaxGenerator


class NonbondedJaxGenerator:
    def __init__(self, ff):
        self.name = "NonbondedForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["sigfix"] = jnp.array([])
        self.paramtree[self.name]["epsfix"] = jnp.array([])

        self.from_force = []
        self.from_residue = []
        self.ra2idx = {}
        self.idx2rai = {}

    def extract(self):
        self.from_residue = self.fftree.get_attribs(
            "NonbondedForce/UseAttributeFromResidue", "name")
        self.from_force = [
            i for i in ["charge", "sigma", "epsilon"]
            if i not in self.from_residue
        ]
        # Build per-atom array for from_force
        for prm in self.from_force:
            vals = self.fftree.get_attribs("NonbondedForce/Atom", prm)
            self.paramtree[self.name][prm] = jnp.array(vals)
        # Build per-atom array for from_residue
        residues = self.fftree.get_nodes("Residues/Residue")
        resvals = {k: [] for k in self.from_residue}
        for resnode in residues:
            resname = resnode.attrs["name"]
            resvals[resname] = []
            atomname = resnode.get_attribs("Atom", "name")
            shift = len(self.ra2idx)
            for natom, aname in enumerate(atomname):
                self.ra2idx[(resname, natom)] = shift + natom
                self.idx2rai[shift + natom] = (resname, atomname, natom)
            for prm in self.from_residue:
                atomval = resnode.get_attribs("Atom", prm)
                resvals[prm].extend(atomval)
        for prm in self.from_residue:
            self.paramtree[self.name][prm] = jnp.array(resvals[prm])
        # Build coulomb14scale and lj14scale
        coulomb14scale, lj14scale = self.fftree.get_attribs(
            "NonbondedForce", ["coulomb14scale", "lj14scale"])[0]
        self.paramtree[self.name]["coulomb14scale"] = jnp.array(
            [coulomb14scale])
        self.paramtree[self.name]["lj14scale"] = jnp.array([lj14scale])

    def overwrite(self):
        # write coulomb14scale
        self.fftree.set_attrib("NonbondedForce", "coulomb14scale",
                               self.paramtree[self.name]["coulomb14scale"])
        # write lj14scale
        self.fftree.set_attrib("NonbondedForce", "lj14scale",
                               self.paramtree[self.name]["lj14scale"])
        # write prm from force
        for prm in self.from_force:
            self.fftree.set_attrib("NonbondedForce/Atom", prm,
                                   self.paramtree[self.name][prm])
        # write prm from residue
        residues = self.fftree.get_nodes("Residues/Residue")
        for prm in self.from_residue:
            vals = self.paramtree[self.name][prm]
            data = []
            for idx in range(vals.shape[0]):
                rname, atomname, aidx = self.idx2rai[idx]
                data.append((rname, aidx, vals[idx]))

            for resnode in residues:
                tmp = sorted(
                    [d for d in data if d[0] == resnode.attrs["name"]],
                    key=lambda x: x[1])
                resnode.set_attrib("Atom", prm, [t[2] for t in tmp])

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):
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
        mscales_coul = mscales_coul.at[2].set(
            self.paramtree[self.name]["coulomb14scale"][0])
        mscales_lj = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # mscale for LJ
        mscales_lj = mscales_lj.at[2].set(
            self.paramtree[self.name]["lj14scale"][0])

        # Coulomb: only support PME for now
        # set PBC
        if nonbondedMethod not in [app.NoCutoff, app.CutoffNonPeriodic]:
            ifPBC = True
        else:
            ifPBC = False

        nbmatcher = TypeMatcher(self.fftree, "NonbondedForce/Atom")
        # load LJ from types
        maps = {}
        for prm in self.from_force:
            maps[prm] = []
            for atom in data.atoms:
                atype = data.atomType[atom]
                ifFound, _, nnode = nbmatcher.matchGeneral([atype])
                if not ifFound:
                    raise DMFFException(
                        "AtomType of %s mismatched in NonbondedForce" %
                        (str(atom)))
                maps[prm].append(nnode)
            maps[prm] = jnp.array(maps[prm], dtype=int)

        for prm in self.from_residue:
            maps[prm] = []
            for atom in data.atoms:
                templateName = self.ff.templateNameForResidue[atom.residue.index]
                aidx = data.atomTemplateIndexes[atom]
                resname, aname = templateName, atom.name
                maps[prm].append(self.ra2idx[(resname, aidx)])
        map_nbfix = []
        map_nbfix = np.array(map_nbfix, dtype=int).reshape((-1, 2))

        self.covalent_map = build_covalent_map(data, 6)

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff
        if "switchDistance" in args and args["switchDistance"] is not None:
            r_switch = args["switchDistance"]
            r_switch = (r_switch if not unit.is_quantity(r_switch) else
                        r_switch.value_in_unit(unit.nanometer))
            ifSwitch = True
        else:
            r_switch = r_cut
            ifSwitch = False

        # PME Settings
        if nonbondedMethod is app.PME:
            a, b, c = system.getDefaultPeriodicBoxVectors()
            box = jnp.array([a._value, b._value, c._value])
            self.ethresh = args.get("ethresh", 1e-6)
            self.coeff_method = args.get("PmeCoeffMethod", "openmm")
            self.fourier_spacing = args.get("PmeSpacing", 0.1)
            kappa, K1, K2, K3 = setup_ewald_parameters(r_cut, self.ethresh,
                                                       box,
                                                       self.fourier_spacing,
                                                       self.coeff_method)

        map_lj = jnp.array(maps["sigma"])
        map_nbfix = jnp.array(map_nbfix)
        map_charge = jnp.array(maps["charge"])

        # Free Energy Settings #
        isFreeEnergy = args.get("isFreeEnergy", False)
        if isFreeEnergy:
            vdwLambda = args.get("vdwLambda", 0.0)
            coulLambda = args.get("coulLambda", 0.0)
            ifStateA = args.get("ifStateA", True)

            # soft-cores
            vdwSoftCore = args.get("vdwSoftCore", False)
            coulSoftCore = args.get("coulSoftCore", False)
            scAlpha = args.get("scAlpha", 0.0)
            scSigma = args.get("scSigma", 0.0)

            # couple
            coupleIndex = args.get("coupleIndex", [])
            if len(coupleIndex) > 0:
                coupleMask = [False for _ in range(len(data.atoms))]
                for atomIndex in coupleIndex:
                    coupleMask[atomIndex] = True
                coupleMask = jnp.array(coupleMask, dtype=bool)
            else:
                coupleMask = None

        if not isFreeEnergy:
            ljforce = LennardJonesForce(r_switch,
                                        r_cut,
                                        map_lj,
                                        map_nbfix,
                                        isSwitch=ifSwitch,
                                        isPBC=ifPBC,
                                        isNoCut=isNoCut)
        else:
            ljforce = LennardJonesFreeEnergyForce(r_switch,
                                                  r_cut,
                                                  map_lj,
                                                  map_nbfix,
                                                  isSwitch=ifSwitch,
                                                  isPBC=ifPBC,
                                                  isNoCut=isNoCut,
                                                  feLambda=vdwLambda,
                                                  coupleMask=coupleMask,
                                                  useSoftCore=vdwSoftCore,
                                                  ifStateA=ifStateA,
                                                  sc_alpha=scAlpha,
                                                  sc_sigma=scSigma)

        ljenergy = ljforce.generate_get_energy()

        # dispersion correction
        useDispersionCorrection = args.get("useDispersionCorrection", False)
        if useDispersionCorrection:
            numTypes = self.paramtree[self.name]["sigma"].shape[0]
            countVec = np.zeros(numTypes, dtype=int)
            countMat = np.zeros((numTypes, numTypes), dtype=int)
            types, count = np.unique(map_lj, return_counts=True)

            for typ, cnt in zip(types, count):
                countVec[typ] += cnt
            for i in range(numTypes):
                for j in range(i, numTypes):
                    if i != j:
                        countMat[i, j] = countVec[i] * countVec[j]
                    else:
                        countMat[i, i] = countVec[i] * (countVec[i] - 1) // 2
            assert np.sum(countMat) == len(map_lj) * (len(map_lj) - 1) // 2

            colv_pairs = np.argwhere(
                np.logical_and(self.covalent_map > 0, self.covalent_map <= 3))
            for pair in colv_pairs:
                if pair[0] <= pair[1]:
                    tmp = (map_lj[pair[0]], map_lj[pair[1]])
                    t1, t2 = min(tmp), max(tmp)
                    countMat[t1, t2] -= 1

            if not isFreeEnergy:
                ljDispCorrForce = LennardJonesLongRangeForce(
                    r_cut, map_lj, map_nbfix, countMat)
            else:
                ljDispCorrForce = LennardJonesLongRangeFreeEnergyForce(
                    r_cut, map_lj, map_nbfix, countMat, vdwLambda, ifStateA,
                    coupleMask)
            ljDispEnergyFn = ljDispCorrForce.generate_get_energy()

        if not isFreeEnergy:
            if nonbondedMethod is not app.PME:
                # do not use PME
                if nonbondedMethod in [
                        app.CutoffPeriodic, app.CutoffNonPeriodic
                ]:
                    # use Reaction Field
                    coulforce = CoulReactionFieldForce(r_cut,
                                                       map_charge,
                                                       isPBC=ifPBC)
                if nonbondedMethod is app.NoCutoff:
                    # use NoCutoff
                    coulforce = CoulNoCutoffForce(map_charge)
            else:
                coulforce = CoulombPMEForce(r_cut, map_charge, kappa,
                                            (K1, K2, K3))
        else:
            assert nonbondedMethod is app.PME, "Only PME is supported in free energy calculations"
            coulforce = CoulombPMEFreeEnergyForce(r_cut,
                                                  map_charge,
                                                  kappa, (K1, K2, K3),
                                                  coulLambda,
                                                  ifStateA=ifStateA,
                                                  coupleMask=coupleMask,
                                                  useSoftCore=coulSoftCore,
                                                  sc_alpha=scAlpha,
                                                  sc_sigma=scSigma)

        coulenergy = coulforce.generate_get_energy()

        if not isFreeEnergy:

            def potential_fn(positions, box, pairs, params):

                # check whether args passed into potential_fn are jnp.array and differentiable
                # note this check will be optimized away by jit
                # it is jit-compatiable
                isinstance_jnp(positions, box, params)

                ljE = ljenergy(positions, box, pairs,
                               params[self.name]["epsilon"],
                               params[self.name]["sigma"],
                               params[self.name]["epsfix"],
                               params[self.name]["sigfix"], mscales_lj)
                coulE = coulenergy(positions, box, pairs,
                                   params[self.name]["charge"], mscales_coul)

                if useDispersionCorrection:
                    ljDispEnergy = ljDispEnergyFn(box,
                                                  params[self.name]['epsilon'],
                                                  params[self.name]['sigma'],
                                                  params[self.name]['epsfix'],
                                                  params[self.name]['sigfix'])

                    return ljE + coulE + ljDispEnergy
                else:
                    return ljE + coulE

            self._jaxPotential = potential_fn
        else:
            # Free Energy
            @jit_condition()
            def potential_fn(positions, box, pairs, params, vdwLambda,
                             coulLambda):
                ljE = ljenergy(positions, box, pairs,
                               params[self.name]["epsilon"],
                               params[self.name]["sigma"],
                               params[self.name]["epsfix"],
                               params[self.name]["sigfix"], mscales_lj,
                               vdwLambda)
                coulE = coulenergy(positions, box, pairs,
                                   params[self.name]["charge"], mscales_coul,
                                   coulLambda)

                if useDispersionCorrection:
                    ljDispEnergy = ljDispEnergyFn(box,
                                                  params[self.name]['epsilon'],
                                                  params[self.name]['sigma'],
                                                  params[self.name]['epsfix'],
                                                  params[self.name]['sigfix'],
                                                  vdwLambda)
                    return ljE + coulE + ljDispEnergy
                else:
                    return ljE + coulE

            self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


dmff.api.jaxGenerators["NonbondedForce"] = NonbondedJaxGenerator


class LennardJonesGenerator:
    def __init__(self, ff):
        self.name = "LennardJonesForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self.paramtree[self.name] = {}
        self.paramtree[self.name]
        self.paramtree[self.name]

    def extract(self):
        for prm in ["sigma", "epsilon"]:
            vals = self.fftree.get_attribs("LennardJonesForce/Atom", prm)
            self.paramtree[self.name][prm] = jnp.array(vals)
            valfix = self.fftree.get_attribs("LennardJonesForce/NBFixPair",
                                             prm)
            self.paramtree[self.name][f"{prm}_nbfix"] = jnp.array(valfix)

        lj14scale = self.fftree.get_attribs("LennardJonesForce",
                                            "lj14scale")[0]
        self.paramtree[self.name]["lj14scale"] = jnp.array([lj14scale])

    def overwrite(self):
        self.fftree.set_attrib("LennardJonesForce", "lj14scale",
                               self.paramtree[self.name]["lj14scale"])
        for prm in ["sigma", "epsilon"]:
            self.fftree.set_attrib("LennardJonesForce/Atom", prm,
                                   self.paramtree[self.name][prm])
            if len(self.paramtree[self.name][f"{prm}_nbfix"]) > 0:
                self.fftree.set_attrib(
                    "LennardJonesForce/NBFixPair", prm,
                    self.paramtree[self.name][f"{prm}_nbfix"])

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):
        methodMap = {
            app.NoCutoff: "NoCutoff",
            app.PME: "CutoffPeriodic",
            app.CutoffPeriodic: "CutoffPeriodic",
            app.CutoffNonPeriodic: "CutoffNonPeriodic"
        }
        if nonbondedMethod not in methodMap:
            raise DMFFException("Illegal nonbonded method for NonbondedForce")
        isNoCut = False
        if nonbondedMethod is app.NoCutoff:
            isNoCut = True

        mscales_lj = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # mscale for LJ
        mscales_lj = mscales_lj.at[2].set(
            self.paramtree[self.name]["lj14scale"][0])

        if nonbondedMethod not in [app.NoCutoff, app.CutoffNonPeriodic]:
            ifPBC = True
        else:
            ifPBC = False

        nbmatcher = TypeMatcher(self.fftree, "LennardJonesForce/Atom")

        maps = {}
        for prm in ["sigma", "epsilon"]:
            maps[prm] = []
            for atom in data.atoms:
                atype = data.atomType[atom]
                ifFound, _, nnode = nbmatcher.matchGeneral([atype])
                if not ifFound:
                    raise DMFFException(
                        "AtomType of %s mismatched in NonbondedForce" %
                        (str(atom)))
                maps[prm].append(nnode)
            maps[prm] = jnp.array(maps[prm], dtype=int)

        map_lj = jnp.array(maps["sigma"])

        ifType = len(self.fftree.get_attribs("LennardJonesForce/Atom",
                                             "type")) != 0
        if ifType:
            atom_labels = self.fftree.get_attribs("LennardJonesForce/Atom",
                                                  "type")
            fix_label1 = self.fftree.get_attribs("LennardJonesForce/NBFixPair",
                                                 "type1")
            fix_label2 = self.fftree.get_attribs("LennardJonesForce/NBFixPair",
                                                 "type2")
        else:
            atom_labels = self.fftree.get_attribs("LennardJonesForce/Atom",
                                                  "class")
            fix_label1 = self.fftree.get_attribs("LennardJonesForce/NBFixPair",
                                                 "class1")
            fix_label2 = self.fftree.get_attribs("LennardJonesForce/NBFixPair",
                                                 "class2")

        map_nbfix = []

        def findIdx(labels, label):
            for ni in range(len(labels)):
                if labels[ni] == label:
                    return ni
            raise DMFFException(
                "AtomType of %s mismatched in LennardJonesForce" % (label))

        for nfix in range(len(fix_label1)):
            l1, l2 = fix_label1[nfix], fix_label2[nfix]
            i1 = findIdx(atom_labels, l1)
            i2 = findIdx(atom_labels, l2)
            map_nbfix.append([i1, i2])
        map_nbfix = np.array(map_nbfix, dtype=int).reshape((-1, 2))
        map_nbfix = jnp.array(map_nbfix)

        colv_map = build_covalent_map(data, 6)

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff
        if "switchDistance" in args and args["switchDistance"] is not None:
            r_switch = args["switchDistance"]
            r_switch = (r_switch if not unit.is_quantity(r_switch) else
                        r_switch.value_in_unit(unit.nanometer))
            ifSwitch = True
        else:
            r_switch = r_cut
            ifSwitch = False

        ljforce = LennardJonesForce(r_switch,
                                    r_cut,
                                    map_lj,
                                    map_nbfix,
                                    colv_map,
                                    isSwitch=ifSwitch,
                                    isPBC=ifPBC,
                                    isNoCut=isNoCut)
        ljenergy = ljforce.generate_get_energy()

        useDispersionCorrection = self.fftree.get_attribs(
            "LennardJonesForce", "useDispersionCorrection")[0] == "True"
        if useDispersionCorrection:
            numTypes = self.paramtree[self.name]["sigma"].shape[0]
            countVec = np.zeros(numTypes, dtype=int)
            countMat = np.zeros((numTypes, numTypes), dtype=int)
            types, count = np.unique(map_lj, return_counts=True)

            for typ, cnt in zip(types, count):
                countVec[typ] += cnt
            for i in range(numTypes):
                for j in range(i, numTypes):
                    if i != j:
                        countMat[i, j] = countVec[i] * countVec[j]
                    else:
                        countMat[i, i] = countVec[i] * (countVec[i] - 1) // 2
            assert np.sum(countMat) == len(map_lj) * (len(map_lj) - 1) // 2

            colv_pairs = np.argwhere(
                np.logical_and(colv_map > 0, colv_map <= 3))
            for pair in colv_pairs:
                if pair[0] <= pair[1]:
                    tmp = (map_lj[pair[0]], map_lj[pair[1]])
                    t1, t2 = min(tmp), max(tmp)
                    countMat[t1, t2] -= 1

            ljDispCorrForce = LennardJonesLongRangeForce(
                r_cut, map_lj, map_nbfix, countMat)

            ljDispEnergyFn = ljDispCorrForce.generate_get_energy()

        def potential_fn(positions, box, pairs, params):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)
            ljE = ljenergy(positions, box, pairs, params[self.name]["epsilon"],
                           params[self.name]["sigma"],
                           params[self.name]["epsilon_nbfix"],
                           params[self.name]["sigma_nbfix"], mscales_lj)

            if useDispersionCorrection:
                ljDispEnergy = ljDispEnergyFn(
                    box, params[self.name]['epsilon'],
                    params[self.name]['sigma'],
                    params[self.name]['epsilon_nbfix'],
                    params[self.name]['sigma_nbfix'])

                return ljE + ljDispEnergy
            else:
                return ljE

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


dmff.api.jaxGenerators["LennardJonesForce"] = LennardJonesGenerator