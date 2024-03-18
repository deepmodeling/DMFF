from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException, isinstance_jnp
from ..admp.pme import setup_ewald_parameters
import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from ..classical.intra import HarmonicBondJaxForce, HarmonicAngleJaxForce, PeriodicTorsionJaxForce, Custom1_5BondJaxForce, CustomTorsionJaxForce
from ..classical.inter import CoulNoCutoffForce, CoulombPMEForce, CoulReactionFieldForce, LennardJonesForce, LennardJonesLongRangeForce, CustomGBForce
from typing import Tuple, List, Union, Callable


class HarmonicBondGenerator:
    """
    A class for generating harmonic bond force field parameters.

    Attributes:
    -----------
    name : str
        The name of the force field.
    ffinfo : dict
        The force field information.
    key_type : str
        The type of the key.
    bond_keys : list of tuple
        The keys of the bonds.
    bond_params : list of tuple
        The parameters of the bonds.
    bond_mask : list of float
        The mask of the bonds.
    _use_smarts : bool
        Whether to use SMARTS.
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        """
        Initializes the HarmonicBondGenerator.

        Parameters:
        -----------
        ffinfo : dict
            The force field information.
        paramset : ParamSet
            The parameter set.
        """
        self.name = "HarmonicBondForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        bond_keys, bond_params, bond_mask = [], [], []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            attribs = node["attrib"]
            if self.key_type is None:
                if "type1" in attribs:
                    self.key_type = "type"
                elif "class1" in attribs:
                    self.key_type = "class"
                else:
                    raise ValueError(
                        "Cannot find key type for HarmonicBondForce.")
            key = (attribs[self.key_type + "1"], attribs[self.key_type + "2"])
            bond_keys.append(key)

            k = float(attribs["k"])
            r0 = float(attribs["length"])
            bond_params.append([k, r0])

            # when the node has mask attribute, it means that the parameter is not trainable.
            # the gradient of this parameter will be zero.
            mask = 1.0
            if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                mask = 0.0
            bond_mask.append(mask)

        self.bond_keys = bond_keys
        bond_length = jnp.array([i[1] for i in bond_params])
        bond_k = jnp.array([i[0] for i in bond_params])
        bond_mask = jnp.array(bond_mask)

        # register parameters to ParamSet
        paramset.addParameter(bond_length, "length",
                              field=self.name, mask=bond_mask)
        # register parameters to ParamSet
        paramset.addParameter(bond_k, "k", field=self.name, mask=bond_mask)

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
        """
        Overwrites the parameter set.

        Parameters:
        -----------
        paramset : ParamSet
            The parameter set.
        """
        bond_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Bond"]

        bond_length = paramset[self.name]["length"]
        bond_k = paramset[self.name]["k"]
        bond_msks = paramset.mask[self.name]["length"]
        for nnode, key in enumerate(self.bond_keys):
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}2"] = key[1]
            r0 = bond_length[nnode]
            k = bond_k[nnode]
            mask = bond_msks[nnode]
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"]["k"] = str(k)
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"]["length"] = str(r0)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                         ]["attrib"]["mask"] = "true"

    def _find_key_index(self, key: Tuple[str, str]) -> int:
        """
        Finds the index of the key.

        Parameters:
        -----------
        key : tuple of str
            The key.

        Returns:
        --------
        int
            The index of the key.
        """
        for i, k in enumerate(self.bond_keys):
            if k[0] == key[0] and k[1] == key[1]:
                return i
            if k[0] == key[1] and k[1] == key[0]:
                return i
        return None

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):
        """
        Creates the potential.

        Parameters:
        -----------
        topdata : DMFFTopology
            The topology data.
        nonbondedMethod : str
            The nonbonded method.
        nonbondedCutoff : float
            The nonbonded cutoff.
        args : list
            The arguments.

        Returns:
        --------
        function
            The potential function.
        """
        # 按照HarmonicBondForce的要求遍历体系中所有的bond，进行匹配
        bond_a1, bond_a2, bond_indices = [], [], []
        for bond in topdata.bonds():
            a1, a2 = bond.atom1, bond.atom2
            i1, i2 = a1.index, a2.index
            if self.key_type == "type":
                key = (a1.meta["type"], a2.meta["type"])
            elif self.key_type == "class":
                key = (a1.meta["class"], a2.meta["class"])
            idx = self._find_key_index(key)
            if idx is None:
                continue
            bond_a1.append(i1)
            bond_a2.append(i2)
            bond_indices.append(idx)
        bond_a1 = jnp.array(bond_a1)
        bond_a2 = jnp.array(bond_a2)
        bond_indices = jnp.array(bond_indices)

        # 创建势函数
        harmonic_bond_force = HarmonicBondJaxForce(
            bond_a1, bond_a2, bond_indices)
        harmonic_bond_energy = harmonic_bond_force.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet, aux=None):
            isinstance_jnp(positions, box, params)
            energy = harmonic_bond_energy(
                positions, box, pairs, params[self.name]["k"], params[self.name]["length"])
            if has_aux:
                return energy, aux
            else:
                return energy

        self._jaxPotential = potential_fn
        return potential_fn


# register the generator
_DMFFGenerators["HarmonicBondForce"] = HarmonicBondGenerator


class HarmonicAngleGenerator:
    """
    A class for generating harmonic angle force field parameters.

    Attributes:
    -----------
    name : str
        The name of the force field.
    ffinfo : dict
        The force field information.
    key_type : str
        The type of the key.
    angle_keys : list of tuple
        The keys of the bonds.
    angle_params : list of tuple
        The parameters of the bonds.
    angle_mask : list of float
        The mask of the bonds.
    _use_smarts : bool
        Whether to use SMARTS.
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        """
        Initializes the HarmonicAngleGenerator.

        Parameters:
        -----------
        ffinfo : dict
            The force field information.
        paramset : ParamSet
            The parameter set.
        """
        self.name = "HarmonicAngleForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        angle_keys, angle_params, angle_mask = [], [], []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            attribs = node["attrib"]

            if self.key_type is None:
                if "type1" in attribs:
                    self.key_type = "type"
                elif "class1" in attribs:
                    self.key_type = "class"
                else:
                    raise ValueError(
                        "Cannot find key type for HarmonicAngleForce.")
            key = (attribs[self.key_type + "1"],
                   attribs[self.key_type + "2"], attribs[self.key_type + "3"])
            angle_keys.append(key)

            k = float(attribs["k"])
            r0 = float(attribs["angle"])
            angle_params.append([k, r0])

            # when the node has mask attribute, it means that the parameter is not trainable.
            # the gradient of this parameter will be zero.
            mask = 1.0
            if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                mask = 0.0
            angle_mask.append(mask)

        self.angle_keys = angle_keys
        angle_theta = jnp.array([i[1] for i in angle_params])
        angle_k = jnp.array([i[0] for i in angle_params])
        angle_mask = jnp.array(angle_mask)

        # register parameters to ParamSet
        paramset.addParameter(angle_theta, "angle",
                              field=self.name, mask=angle_mask)
        # register parameters to ParamSet
        paramset.addParameter(angle_k, "k", field=self.name, mask=angle_mask)

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
        """
        Overwrites the parameter set.

        Parameters:
        -----------
        paramset : ParamSet
            The parameter set.
        """
        angle_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Angle"]

        angle_theta = paramset[self.name]["angle"]
        angle_k = paramset[self.name]["k"]
        angle_msks = paramset.mask[self.name]["angle"]
        for nnode, key in enumerate(self.angle_keys):
            self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}2"] = key[1]
            self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}3"] = key[2]
            theta0 = angle_theta[nnode]
            k = angle_k[nnode]
            mask = angle_msks[nnode]
            self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]
                                                     ]["attrib"]["k"] = str(k)
            self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]
                                                     ]["attrib"]["angle"] = str(theta0)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][angle_node_indices[nnode]
                                                         ]["attrib"]["mask"] = "true"

    def _find_key_index(self, key: Tuple[str, str]) -> int:
        """
        Finds the index of the key.

        Parameters:
        -----------
        key : tuple of str
            The key.

        Returns:
        --------
        int
            The index of the key.
        """
        for i, k in enumerate(self.angle_keys):
            if k[0] == key[0] and k[1] == key[1] and k[2] == key[2]:
                return i
            if k[0] == key[2] and k[1] == key[1] and k[2] == key[0]:
                return i
        return None

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):
        """
        Creates the potential.

        Parameters:
        -----------
        topdata : DMFFTopology
            The topology data.
        nonbondedMethod : str
            The nonbonded method.
        nonbondedCutoff : float
            The nonbonded cutoff.
        args : list
            The arguments.

        Returns:
        --------
        function
            The potential function.
        """
        angle_a1, angle_a2, angle_a3, angle_indices = [], [], [], []
        angles = []
        acenters = {}
        for bond in topdata.bonds():
            a1, a2 = bond.atom1, bond.atom2
            i1, i2 = a1.index, a2.index
            if i1 not in acenters:
                acenters[i1] = [a1]
            acenters[i1].append(a2)
            if i2 not in acenters:
                acenters[i2] = [a2]
            acenters[i2].append(a1)
        for icenter in acenters:
            if len(acenters[icenter]) < 3:
                continue
            acenter = acenters[icenter][0]
            alinks = acenters[icenter][1:]
            for ii in range(len(alinks)):
                for jj in range(ii+1, len(alinks)):
                    angles.append((alinks[ii], acenter, alinks[jj]))
        for angle in angles:
            a1, a2, a3 = angle
            i1, i2, i3 = a1.index, a2.index, a3.index
            if self.key_type == "type":
                key = (a1.meta["type"], a2.meta["type"], a3.meta["type"])
            elif self.key_type == "class":
                key = (a1.meta["class"], a2.meta["class"], a3.meta["class"])
            idx = self._find_key_index(key)
            if idx is None:
                continue
            angle_a1.append(i1)
            angle_a2.append(i2)
            angle_a3.append(i3)
            angle_indices.append(idx)
        angle_a1 = jnp.array(angle_a1)
        angle_a2 = jnp.array(angle_a2)
        angle_a3 = jnp.array(angle_a3)
        angle_indices = jnp.array(angle_indices)

        harmonic_angle_force = HarmonicAngleJaxForce(
            angle_a1, angle_a2, angle_a3, angle_indices)
        harmonic_angle_energy = harmonic_angle_force.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet, aux=None):
            isinstance_jnp(positions, box, params)
            energy = harmonic_angle_energy(
                positions, box, pairs, params[self.name]["k"], params[self.name]["angle"])
            if has_aux:
                return energy, aux
            else:
                return energy

        self._jaxPotential = potential_fn
        return potential_fn


# register the generator
_DMFFGenerators["HarmonicAngleForce"] = HarmonicAngleGenerator


class PeriodicTorsionGenerator:

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        """
        Initializes a PeriodicTorsionForce object.

        Args:
        - ffinfo (dict): A dictionary containing force field information.
        - paramset (ParamSet): A ParamSet object to register parameters.

        Raises:
        - ValueError: If the ordering of PeriodicTorsionForce is not "amber".

        Returns:
        - None
        """
        self.name = "PeriodicTorsionForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self._use_smarts = False
        self.key_type = None

        if "ordering" in self.ffinfo["Forces"][self.name] and self.ffinfo["Forces"][self.name]["ordering"] != "amber":
            raise ValueError("PeriodicTorsionForce ordering must be amber")

        proper_keys, proper_periods, proper_prms = [], [], []
        proper_key_to_prms = {}
        improper_keys, improper_periods, improper_prms = [], [], []
        improper_key_to_prms = {}
        for node in self.ffinfo["Forces"][self.name]["node"]:
            attribs = node["attrib"]
            if "type1" in attribs:
                self.key_type = "type"
            elif "class1" in attribs:
                self.key_type = "class"
            key = (attribs[self.key_type + "1"], attribs[self.key_type + "2"],
                   attribs[self.key_type + "3"], attribs[self.key_type + "4"])
            if node["name"] == "Proper":
                proper_keys.append(key)
            elif node["name"] == "Improper":
                improper_keys.append(key)

            mask = 1.0
            if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                mask = 0.0

            for period_key in attribs.keys():
                if "periodicity" not in period_key:
                    continue
                order = int(period_key.replace("periodicity", ""))
                period = int(attribs[period_key])
                phase = float(attribs["phase" + str(order)])
                k = float(attribs["k" + str(order)])
                if node["name"] == "Proper":
                    proper_periods.append(period)
                    proper_prms.append([phase, k, mask])
                    if len(proper_keys) - 1 not in proper_key_to_prms:
                        proper_key_to_prms[len(proper_keys) - 1] = []
                    proper_key_to_prms[len(
                        proper_keys) - 1].append(len(proper_periods) - 1)
                elif node["name"] == "Improper":
                    improper_periods.append(period)
                    improper_prms.append([phase, k, mask])
                    if len(improper_keys) - 1 not in improper_key_to_prms:
                        improper_key_to_prms[len(improper_keys) - 1] = []
                    improper_key_to_prms[len(
                        improper_keys) - 1].append(len(improper_periods) - 1)

        self.proper_keys = proper_keys
        self.proper_periods = jnp.array(proper_periods)
        self.proper_key_to_prms = proper_key_to_prms
        proper_phase = jnp.array([i[0] for i in proper_prms])
        proper_k = jnp.array([i[1] for i in proper_prms])
        proper_mask = jnp.array([i[2] for i in proper_prms])
        # register parameters to ParamSet
        paramset.addParameter(proper_phase, "proper_phase",
                              field=self.name, mask=proper_mask)
        paramset.addParameter(proper_k, "proper_k",
                              field=self.name, mask=proper_mask)

        self.imp_keys = improper_keys
        self.imp_periods = jnp.array(improper_periods)
        self.imp_key_to_prms = improper_key_to_prms
        improper_phase = jnp.array([i[0] for i in improper_prms])
        improper_k = jnp.array([i[1] for i in improper_prms])
        improper_mask = jnp.array([i[2] for i in improper_prms])
        # register parameters to ParamSet
        paramset.addParameter(improper_phase, "improper_phase",
                              field=self.name, mask=improper_mask)
        paramset.addParameter(improper_k, "improper_k",
                              field=self.name, mask=improper_mask)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        # paramset to ffinfo
        proper_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Proper"]
        improper_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Improper"]

        proper_phase = paramset[self.name]["proper_phase"]
        proper_k = paramset[self.name]["proper_k"]
        proper_msks = paramset.mask[self.name]["proper"]
        for nnode, key in enumerate(self.proper_keys):
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}2"] = key[1]
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}3"] = key[2]
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}4"] = key[3]
            for nitem, item in enumerate(self.proper_key_to_prms[nnode]):
                phase, k = proper_phase[item], proper_k[item]
                mask = proper_msks[item]
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                         ]["attrib"]["periodicity" + str(nitem + 1)] = str(self.proper_periods[item])
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                         ]["attrib"]["phase" + str(nitem + 1)] = str(phase)
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                         ]["attrib"]["k" + str(nitem + 1)] = str(k)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                                                         ]["attrib"]["mask"] = "true"

        improper_phase = paramset[self.name]["improper_phase"]
        improper_k = paramset[self.name]["improper_k"]
        improper_msks = paramset.mask[self.name]["improper"]
        for nnode, key in enumerate(self.imp_keys):
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}2"] = key[1]
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}3"] = key[2]
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}4"] = key[3]
            for nitem, item in enumerate(self.imp_key_to_prms[nnode]):
                phase = improper_phase[item]
                k = improper_k[item]
                mask = improper_msks[item]
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                         ]["attrib"]["periodicity" + str(nitem + 1)] = str(self.imp_periods[item])
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                         ]["attrib"]["phase" + str(nitem + 1)] = str(phase)
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                         ]["attrib"]["k" + str(nitem + 1)] = str(k)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                                                         ]["attrib"]["mask"] = "true"

    def _find_proper_key_index(self, key: Tuple[str, str, str, str]) -> int:
        wc_patch = []
        for i, k in enumerate(self.proper_keys):
            if k[0] in ["", key[0]] and k[1] in ["", key[1]] and k[2] in ["", key[2]] and k[3] in ["", key[3]]:
                if "" in k:
                    wc_patch.append(i)
                else:
                    return i
            if k[0] in ["", key[3]] and k[1] in ["", key[2]] and k[2] in ["", key[1]] and k[3] in ["", key[0]]:
                if "" in k:
                    wc_patch.append(i)
                else:
                    return i
        if len(wc_patch) > 0:
            return wc_patch[0]
        return None

    def _find_improper_key_index(self, improper):
        
        type1 = improper[0].meta[self.key_type]
        type2 = improper[1].meta[self.key_type]
        type3 = improper[2].meta[self.key_type]
        type4 = improper[3].meta[self.key_type]
        
        def _wild_match(tp, tps):
            if tps == "":
                return True
            if tp == tps:
                return True
            return False

        matched = None
        for ndef, tordef in enumerate(self.imp_keys):
            types1 = tordef[0]
            types2 = tordef[1]
            types3 = tordef[2]
            types4 = tordef[3]
            hasWildcard = ("" in (types1, types2, types3, types4))

            if matched is not None and hasWildcard:
                continue

            import itertools
            if type1 in types1:
                for (t2, t3, t4) in itertools.permutations(((type2, 1), (type3, 2), (type4, 3))):
                    if _wild_match(t2[0], types2) and _wild_match(t3[0], types3) and _wild_match(t4[0], types4):
                        a1 = improper[t2[1]].index
                        a2 = improper[t3[1]].index
                        e1 = improper[t2[1]].element
                        e2 = improper[t3[1]].element
                        m1 = app.element.get_by_symbol(e1).mass
                        m2 = app.element.get_by_symbol(e2).mass
                        if e1 == e2 and a1 > a2:
                            (a1, a2) = (a2, a1)
                        elif e1 != "C" and (e2 == "C" or m1 < m2):
                            (a1, a2) = (a2, a1)
                        matched = (a1, a2, improper[0].index, improper[t4[1]].index, ndef)
                        break
        if matched is None:
            return None, None
        return matched[4], matched[:4]


    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):
        
        if self.key_type is None:
            def potential_fn_zero(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet) -> jnp.ndarray:
                return jnp.zeros((1,))
            self._jaxPotential = potential_fn_zero
            return potential_fn_zero

        proper_list = []

        acenters = {}
        atoms = [a for a in topdata.atoms()]
        for bond in topdata.bonds():
            a1, a2 = bond.atom1, bond.atom2
            i1, i2 = a1.index, a2.index
            if i1 not in acenters:
                acenters[i1] = []
            acenters[i1].append(i2)
            if i2 not in acenters:
                acenters[i2] = []
            acenters[i2].append(i1)

        # find rotamers and loop over proper torsions on the rotamer
        for bond in topdata.bonds():
            a1, a2 = bond.atom1, bond.atom2
            i1, i2 = a1.index, a2.index
            alinks1 = [i for i in acenters[i1] if i != i2]
            alinks2 = [i for i in acenters[i2] if i != i1]
            for i3 in alinks1:
                for i4 in alinks2:
                    if i3 != i4:
                        proper_list.append(
                            (atoms[i3], atoms[i1], atoms[i2], atoms[i4]))

        impr_list = []
        # find atoms that link with three other atoms
        import itertools as it
        for i1 in acenters:
            if len(acenters[i1]) < 3:
                continue
            for item in it.combinations(acenters[i1], 3):
                impr_list.append(
                    (atoms[i1], atoms[item[0]], atoms[item[1]], atoms[item[2]]))

        # create potential
        proper_a1, proper_a2, proper_a3, proper_a4, proper_indices, proper_period = [
        ], [], [], [], [], []
        for proper in proper_list:
            pidx = self._find_proper_key_index(
                (proper[0].meta[self.key_type], proper[1].meta[self.key_type], proper[2].meta[self.key_type], proper[3].meta[self.key_type]))
            if pidx is None:
                continue

            prm_indices = self.proper_key_to_prms[pidx]
            for prm_idx in prm_indices:
                prm_period = self.proper_periods[prm_idx]
                proper_a1.append(proper[0].index)
                proper_a2.append(proper[1].index)
                proper_a3.append(proper[2].index)
                proper_a4.append(proper[3].index)
                proper_indices.append(prm_idx)
                proper_period.append(prm_period)

        proper_a1 = jnp.array(proper_a1)
        proper_a2 = jnp.array(proper_a2)
        proper_a3 = jnp.array(proper_a3)
        proper_a4 = jnp.array(proper_a4)
        proper_indices = jnp.array(proper_indices)
        proper_period = jnp.array(proper_period)

        improper_a1, improper_a2, improper_a3, improper_a4, improper_indices, improper_period = [], [], [], [], [], []
        for improper in impr_list:
            iidx, order = self._find_improper_key_index(improper)
            if iidx is None:
                continue

            prm_indices = self.imp_key_to_prms[iidx]
            for prm_idx in prm_indices:
                prm_period = self.imp_periods[prm_idx]
                improper_a1.append(atoms[order[0]].index)
                improper_a2.append(atoms[order[1]].index)
                improper_a3.append(atoms[order[2]].index)
                improper_a4.append(atoms[order[3]].index)
                improper_indices.append(prm_idx)
                improper_period.append(prm_period)
        improper_a1 = jnp.array(improper_a1)
        improper_a2 = jnp.array(improper_a2)
        improper_a3 = jnp.array(improper_a3)
        improper_a4 = jnp.array(improper_a4)
        improper_indices = jnp.array(improper_indices)
        improper_period = jnp.array(improper_period)

        proper_func = PeriodicTorsionJaxForce(
            proper_a1, proper_a2, proper_a3, proper_a4, proper_indices, proper_period)
        proper_energy = proper_func.generate_get_energy()
        improper_func = PeriodicTorsionJaxForce(
            improper_a1, improper_a2, improper_a3, improper_a4, improper_indices, improper_period)
        improper_energy = improper_func.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet, aux=None):
            isinstance_jnp(positions, box, params)
            proper_energy_ = proper_energy(
                positions, box, pairs, params[self.name]["proper_k"], params[self.name]["proper_phase"])
            improper_energy_ = improper_energy(
                positions, box, pairs, params[self.name]["improper_k"], params[self.name]["improper_phase"])
            if has_aux:
                return proper_energy_ + improper_energy_, aux
            else:
                return proper_energy_ + improper_energy_

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["PeriodicTorsionForce"] = PeriodicTorsionGenerator


class CustomTorsionGenerator:

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        """
        Initializes a PeriodicTorsionForce object.

        Args:
        - ffinfo (dict): A dictionary containing force field information.
        - paramset (ParamSet): A ParamSet object to register parameters.

        Returns:
        - None
        """
        self.name = "CustomTorsionForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self._use_smarts = False
        self.key_type = None
        self.torsionIndices = [i for i in range(len(self.ffinfo["Forces"][self.name]["node"])) if "roper" in self.ffinfo["Forces"][self.name]["node"][i]["name"]]

        proper_keys, proper_periods, proper_prms, proper_shift = [], [], [], []
        proper_key_to_prms = {}
        improper_keys, improper_periods, improper_prms, improper_shift = [], [], [], []
        improper_key_to_prms = {}
        for i in self.torsionIndices:
            node = self.ffinfo["Forces"][self.name]["node"][i]
            attribs = node["attrib"]
            if "type1" in attribs:
                self.key_type = "type"
            elif "class1" in attribs:
                self.key_type = "class"
            key = (attribs[self.key_type + "1"], attribs[self.key_type + "2"],
                   attribs[self.key_type + "3"], attribs[self.key_type + "4"])
            if node["name"] == "Proper":
                proper_keys.append(key)
            elif node["name"] == "Improper":
                improper_keys.append(key)

            mask = 1.0
            if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                mask = 0.0

            for period_key in attribs.keys():
                if "per" not in period_key:
                    continue
                order = int(period_key.replace("per", ""))
                period = int(attribs[period_key])
                phase = float(attribs["phase" + str(order)])
                k = float(attribs["k" + str(order)])
                shift = float(attribs["shift"])/4
                if node["name"] == "Proper":
                    proper_periods.append(period)
                    proper_prms.append([phase, k, mask, shift])
                    if len(proper_keys) - 1 not in proper_key_to_prms:
                        proper_key_to_prms[len(proper_keys) - 1] = []
                    proper_key_to_prms[len(
                        proper_keys) - 1].append(len(proper_periods) - 1)
                elif node["name"] == "Improper":
                    improper_periods.append(period)
                    improper_prms.append([phase, k, mask, shift])
                    if len(improper_keys) - 1 not in improper_key_to_prms:
                        improper_key_to_prms[len(improper_keys) - 1] = []
                    improper_key_to_prms[len(
                        improper_keys) - 1].append(len(improper_periods) - 1)

        self.proper_keys = proper_keys
        self.proper_periods = jnp.array(proper_periods)
        self.proper_key_to_prms = proper_key_to_prms
        proper_phase = jnp.array([i[0] for i in proper_prms])
        proper_k = jnp.array([i[1] for i in proper_prms])
        proper_mask = jnp.array([i[2] for i in proper_prms])
        proper_shift = jnp.array([i[3] for i in proper_prms])
        # register parameters to ParamSet
        paramset.addParameter(proper_phase, "proper_phase",
                              field=self.name, mask=proper_mask)
        paramset.addParameter(proper_k, "proper_k",
                              field=self.name, mask=proper_mask)
        paramset.addParameter(proper_shift, "proper_shift",
                              field=self.name, mask=proper_mask)

        self.imp_keys = improper_keys
        self.imp_periods = jnp.array(improper_periods)
        self.imp_key_to_prms = improper_key_to_prms
        improper_phase = jnp.array([i[0] for i in improper_prms])
        improper_k = jnp.array([i[1] for i in improper_prms])
        improper_mask = jnp.array([i[2] for i in improper_prms])
        improper_shift = jnp.array([i[3] for i in improper_prms])
        # register parameters to ParamSet
        paramset.addParameter(improper_phase, "improper_phase",
                              field=self.name, mask=improper_mask)
        paramset.addParameter(improper_k, "improper_k",
                              field=self.name, mask=improper_mask)
        paramset.addParameter(improper_shift, "improper_shift",
                              field=self.name, mask=improper_mask)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        # paramset to ffinfo
        proper_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if
                               self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Proper"]
        improper_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if
                                 self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Improper"]

        proper_phase = paramset[self.name]["proper_phase"]
        proper_k = paramset[self.name]["proper_k"]
        proper_shift = paramset[self.name]["proper_shift"]
        proper_msks = paramset.mask[self.name]["proper_phase"]
        for nnode, key in enumerate(self.proper_keys):
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}2"] = key[1]
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}3"] = key[2]
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}4"] = key[3]
            shiftTem = 0
            for nitem, item in enumerate(self.proper_key_to_prms[nnode]):
                phase, k, shift = proper_phase[item], proper_k[item], proper_shift[item]
                mask = proper_msks[item]
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                ]["attrib"]["per" + str(nitem + 1)] = str(self.proper_periods[item])
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                ]["attrib"]["phase" + str(nitem + 1)] = str(phase)
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                ]["attrib"]["k" + str(nitem + 1)] = str(k)
                shiftTem += shift
            self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
            ]["attrib"]["shift"] = str(shiftTem)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][proper_node_indices[nnode]
                ]["attrib"]["mask"] = "true"

        improper_phase = paramset[self.name]["improper_phase"]
        improper_k = paramset[self.name]["improper_k"]
        improper_shift = paramset[self.name]["improper_shift"]
        improper_msks = paramset.mask[self.name]["improper_phase"]
        for nnode, key in enumerate(self.imp_keys):
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}2"] = key[1]
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}3"] = key[2]
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
            ]["attrib"][f"{self.key_type}4"] = key[3]
            shiftTem = 0
            for nitem, item in enumerate(self.imp_key_to_prms[nnode]):
                phase = improper_phase[item]
                k = improper_k[item]
                shift = improper_shift[item]
                mask = improper_msks[item]
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                ]["attrib"]["per" + str(nitem + 1)] = str(self.imp_periods[item])
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                ]["attrib"]["phase" + str(nitem + 1)] = str(phase)
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                ]["attrib"]["k" + str(nitem + 1)] = str(k)
                shiftTem += shift
            self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
            ]["attrib"]["shift"] = str(shiftTem)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][improper_node_indices[nnode]
                ]["attrib"]["mask"] = "true"

    def _find_proper_key_index(self, key: Tuple[str, str, str, str]) -> int:
        wc_patch = []
        for i, k in enumerate(self.proper_keys):
            if k[0] in ["", key[0]] and k[1] in ["", key[1]] and k[2] in ["", key[2]] and k[3] in ["", key[3]]:
                if "" in k:
                    wc_patch.append(i)
                else:
                    return i
            if k[0] in ["", key[3]] and k[1] in ["", key[2]] and k[2] in ["", key[1]] and k[3] in ["", key[0]]:
                if "" in k:
                    wc_patch.append(i)
                else:
                    return i
        if len(wc_patch) > 0:
            return wc_patch[0]
        return None

    def _find_improper_key_index(self, improper):

        type1 = improper[0].meta[self.key_type]
        type2 = improper[1].meta[self.key_type]
        type3 = improper[2].meta[self.key_type]
        type4 = improper[3].meta[self.key_type]

        def _wild_match(tp, tps):
            if tps == "":
                return True
            if tp == tps:
                return True
            return False

        matched = None
        for ndef, tordef in enumerate(self.imp_keys):
            types1 = tordef[0]
            types2 = tordef[1]
            types3 = tordef[2]
            types4 = tordef[3]
            hasWildcard = ("" in (types1, types2, types3, types4))

            if matched is not None and hasWildcard:
                continue

            import itertools
            if type1 in types1:
                for (t2, t3, t4) in itertools.permutations(((type2, 1), (type3, 2), (type4, 3))):
                    if _wild_match(t2[0], types2) and _wild_match(t3[0], types3) and _wild_match(t4[0], types4):
                        a1 = improper[t2[1]].index
                        a2 = improper[t3[1]].index
                        e1 = improper[t2[1]].element
                        e2 = improper[t3[1]].element
                        m1 = app.element.get_by_symbol(e1).mass
                        m2 = app.element.get_by_symbol(e2).mass
                        if e1 == e2 and a1 > a2:
                            (a1, a2) = (a2, a1)
                        elif e1 != "C" and (e2 == "C" or m1 < m2):
                            (a1, a2) = (a2, a1)
                        matched = (a1, a2, improper[0].index, improper[t4[1]].index, ndef)
                        break
        if matched is None:
            return None, None
        return matched[4], matched[:4]

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):

        if self.key_type is None:
            def potential_fn_zero(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray,
                                  params: ParamSet) -> jnp.ndarray:
                return jnp.zeros((1,))

            self._jaxPotential = potential_fn_zero
            return potential_fn_zero

        proper_list = []

        acenters = {}
        atoms = [a for a in topdata.atoms()]
        for bond in topdata.bonds():
            a1, a2 = bond.atom1, bond.atom2
            i1, i2 = a1.index, a2.index
            if i1 not in acenters:
                acenters[i1] = []
            acenters[i1].append(i2)
            if i2 not in acenters:
                acenters[i2] = []
            acenters[i2].append(i1)

        # find rotamers and loop over proper torsions on the rotamer
        for bond in topdata.bonds():
            a1, a2 = bond.atom1, bond.atom2
            i1, i2 = a1.index, a2.index
            alinks1 = [i for i in acenters[i1] if i != i2]
            alinks2 = [i for i in acenters[i2] if i != i1]
            for i3 in alinks1:
                for i4 in alinks2:
                    if i3 != i4:
                        proper_list.append(
                            (atoms[i3], atoms[i1], atoms[i2], atoms[i4]))

        impr_list = []
        # find atoms that link with three other atoms
        import itertools as it
        for i1 in acenters:
            if len(acenters[i1]) < 3:
                continue
            for item in it.combinations(acenters[i1], 3):
                impr_list.append(
                    (atoms[i1], atoms[item[0]], atoms[item[1]], atoms[item[2]]))

        # create potential
        proper_a1, proper_a2, proper_a3, proper_a4, proper_indices, proper_period = [
        ], [], [], [], [], []
        for proper in proper_list:
            pidx = self._find_proper_key_index(
                (proper[0].meta[self.key_type], proper[1].meta[self.key_type], proper[2].meta[self.key_type],
                 proper[3].meta[self.key_type]))
            if pidx is None:
                continue

            prm_indices = self.proper_key_to_prms[pidx]
            for prm_idx in prm_indices:
                prm_period = self.proper_periods[prm_idx]
                proper_a1.append(proper[0].index)
                proper_a2.append(proper[1].index)
                proper_a3.append(proper[2].index)
                proper_a4.append(proper[3].index)
                proper_indices.append(prm_idx)
                proper_period.append(prm_period)

        proper_a1 = jnp.array(proper_a1)
        proper_a2 = jnp.array(proper_a2)
        proper_a3 = jnp.array(proper_a3)
        proper_a4 = jnp.array(proper_a4)
        proper_indices = jnp.array(proper_indices)
        proper_period = jnp.array(proper_period)

        improper_a1, improper_a2, improper_a3, improper_a4, improper_indices, improper_period = [], [], [], [], [], []
        for improper in impr_list:
            iidx, order = self._find_improper_key_index(improper)
            if iidx is None:
                continue

            prm_indices = self.imp_key_to_prms[iidx]
            for prm_idx in prm_indices:
                prm_period = self.imp_periods[prm_idx]
                improper_a1.append(atoms[order[0]].index)
                improper_a2.append(atoms[order[1]].index)
                improper_a3.append(atoms[order[2]].index)
                improper_a4.append(atoms[order[3]].index)
                improper_indices.append(prm_idx)
                improper_period.append(prm_period)
        improper_a1 = jnp.array(improper_a1)
        improper_a2 = jnp.array(improper_a2)
        improper_a3 = jnp.array(improper_a3)
        improper_a4 = jnp.array(improper_a4)
        improper_indices = jnp.array(improper_indices)
        improper_period = jnp.array(improper_period)

        proper_func = CustomTorsionJaxForce(
            proper_a1, proper_a2, proper_a3, proper_a4, proper_indices, proper_period)
        proper_energy = proper_func.generate_get_energy()
        improper_func = CustomTorsionJaxForce(
            improper_a1, improper_a2, improper_a3, improper_a4, improper_indices, improper_period)
        improper_energy = improper_func.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet, aux=None):
            isinstance_jnp(positions, box, params)
            proper_energy_ = proper_energy(
                positions, box, pairs, params[self.name]["proper_k"], params[self.name]["proper_phase"], params[self.name]["proper_shift"])
            improper_energy_ = improper_energy(
                positions, box, pairs, params[self.name]["improper_k"], params[self.name]["improper_phase"], params[self.name]["improper_shift"])
            if has_aux:
                return proper_energy_ + improper_energy_, aux
            else:
                return proper_energy_ + improper_energy_

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["CustomTorsionForce"] = CustomTorsionGenerator


class NonbondedGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "NonbondedForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.coulomb14scale = float(
            self.ffinfo["Forces"]["NonbondedForce"]["meta"].get("coulomb14scale", 0.8333333333333334))
        self.lj14scale = float(
            self.ffinfo["Forces"]["NonbondedForce"]["meta"].get("lj14scale", 0.5))
        self.key_type = None
        self.type_to_charge = {}
        
        self.charge_in_residue = False
        for node in self.ffinfo["Forces"]["NonbondedForce"]["node"]:
            if not self.charge_in_residue and node["name"] == "UseAttributeFromResidue":
                if node["attrib"]["name"] == "charge":
                    self.charge_in_residue = True
        
        types, sigma, epsilon, atom_mask = [], [], [], []
        for node in self.ffinfo["Forces"]["NonbondedForce"]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                self.key_type = None
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                types.append(attribs[self.key_type])
                sigma.append(float(attribs["sigma"]))
                epsilon.append(float(attribs["epsilon"]))
                mask = 1.0
                if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                    mask = 0.0
                atom_mask.append(mask)
                if not self.charge_in_residue:
                    if "charge" not in attribs:
                        raise ValueError("No charge information found in NonbondedForce or Residues.")
                    self.type_to_charge[attribs[self.key_type]] = float(attribs["charge"])

        sigma = jnp.array(sigma)
        epsilon = jnp.array(epsilon)
        atom_mask = jnp.array(atom_mask)
        self.atom_keys = types
        paramset.addParameter(sigma, "sigma", field=self.name, mask=atom_mask)
        paramset.addParameter(epsilon, "epsilon", field=self.name, mask=atom_mask)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        sigma = paramset[self.name]["sigma"]
        epsilon = paramset[self.name]["epsilon"]
        atom_mask = paramset.mask[self.name]["sigma"]

        node2atom = [i for i in range(len(self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Atom"]

        for natom in range(len(self.atom_keys)):
            nnode = node2atom[natom]
            sig_new = sigma[natom]
            eps_new = epsilon[natom]
            mask = atom_mask[natom]
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["sigma"] = str(sig_new)
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["epsilon"] = str(eps_new)
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["mask"] = "true"

    def _find_atype_key_index(self, atype: str):
        for n, i in enumerate(self.atom_keys):
            if i == atype:
                return n
        return None
    
    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):
        methodMap = {
            app.NoCutoff: "NoCutoff",
            app.CutoffPeriodic: "CutoffPeriodic",
            app.CutoffNonPeriodic: "CutoffNonPeriodic",
            app.PME: "PME",
        }
        methodString = methodMap[nonbondedMethod]
        if nonbondedMethod not in methodMap:
            raise DMFFException("Illegal nonbonded method for NonbondedForce")

        isNoCut = False
        if nonbondedMethod is app.NoCutoff:
            isNoCut = True

        mscales_coul = jnp.array([0.0, 0.0, self.coulomb14scale, 1.0, 1.0,
                                  1.0])
        mscales_lj = jnp.array([0.0, 0.0, self.lj14scale, 1.0, 1.0,
                                1.0])

        # coulomb
        # set PBC
        if nonbondedMethod not in [app.NoCutoff, app.CutoffNonPeriodic]:
            ifPBC = True
        else:
            ifPBC = False

        if self.charge_in_residue:
            charges = [a.meta["charge"] for a in topdata.atoms()]
            charges = jnp.array(charges)
        else:
            types = [a.meta[self.key_type] for a in topdata.atoms()]
            charges = jnp.array([self.type_to_charge[i] for i in types])

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff

        # PME Settings
        if nonbondedMethod is app.PME:
            cell = topdata.getPeriodicBoxVectors()
            self.ethresh = kwargs.get("ethresh", 1e-6)
            self.coeff_method = kwargs.get("PmeCoeffMethod", "openmm")
            self.fourier_spacing = kwargs.get("PmeSpacing", 0.1)
            kappa, K1, K2, K3 = setup_ewald_parameters(r_cut, self.ethresh,
                                                       cell,
                                                       self.fourier_spacing,
                                                       self.coeff_method)
        if nonbondedMethod is not app.PME:
            # do not use PME
            if nonbondedMethod in [app.CutoffPeriodic, app.CutoffNonPeriodic]:
                # use Reaction Field
                coulforce = CoulReactionFieldForce(r_cut, charges, isPBC=ifPBC)
            if nonbondedMethod is app.NoCutoff:
                # use NoCutoff
                coulforce = CoulNoCutoffForce(init_charges=charges)
        else:
            coulforce = CoulombPMEForce(r_cut, charges, kappa, (K1, K2, K3))
        
        self.pme_force = coulforce
        coulenergy = coulforce.generate_get_energy()

        # LJ
        atypes = [a.meta[self.key_type] for a in topdata.atoms()]
        map_prm = []
        for atype in atypes:
            pidx = self._find_atype_key_index(atype)
            if pidx is None:
                raise DMFFException(f"Atom type {atype} not found.")
            map_prm.append(pidx)
        map_prm = jnp.array(map_prm)

        # not use nbfix for now
        map_nbfix = []
        map_nbfix = jnp.array(map_nbfix, dtype=int).reshape((-1, 3))
        eps_nbfix = jnp.array(map_nbfix, dtype=float).reshape((-1, 3))
        sig_nbfix = jnp.array(map_nbfix, dtype=float).reshape((-1, 3))

        if methodString in ["NoCutoff", "CutoffNonPeriodic"]:
            isPBC = False
            if methodString == "NoCutoff":
                isNoCut = True
            else:
                isNoCut = False
        else:
            isPBC = True
            isNoCut = False

        ljforce = LennardJonesForce(0.0,
                                    r_cut,
                                    map_prm,
                                    map_nbfix,
                                    isSwitch=False,
                                    isPBC=isPBC,
                                    isNoCut=isNoCut)
        ljenergy = ljforce.generate_get_energy()

        # dispersion correction
        use_disp_corr = False
        if "useDispersionCorrection" in kwargs and kwargs["useDispersionCorrection"]:
            use_disp_corr = True
            numTypes = len(self.atom_keys)
            countVec = np.zeros(numTypes, dtype=int)
            countMat = np.zeros((numTypes, numTypes), dtype=int)
            types, count = np.unique(map_prm, return_counts=True)
            for typ, cnt in zip(types, count):
                countVec[typ] += cnt
            for i in range(numTypes):
                for j in range(i, numTypes):
                    if i != j:
                        countMat[i, j] = countVec[i] * countVec[j]
                    else:
                        countMat[i, j] = countVec[i] * (countVec[i] - 1) // 2
            assert np.sum(countMat) == len(map_prm) * (len(map_prm) - 1) // 2

            coval_map = topdata.buildCovMat()
            colv_pairs = np.argwhere(
                np.logical_and(coval_map > 0, coval_map <= 3))
            for pair in colv_pairs:
                if pair[0] <= pair[1]:
                    tmp = (map_prm[pair[0]], map_prm[pair[1]])
                    t1, t2 = min(tmp), max(tmp)
                    countMat[t1, t2] -= 1

            ljDispCorrForce = LennardJonesLongRangeForce(r_cut, map_prm, map_nbfix, countMat)
            ljDispEnergyFn = ljDispCorrForce.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux=None):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)

            coulE = coulenergy(positions, box, pairs, mscales_coul)
            
            ljE = ljenergy(positions, box, pairs, params[self.name]["epsilon"],
                            params[self.name]["sigma"], eps_nbfix, sig_nbfix, mscales_lj)
            if use_disp_corr:
                ljdispE = ljDispEnergyFn(box, params[self.name]["epsilon"],
                            params[self.name]["sigma"], eps_nbfix, sig_nbfix)
                if has_aux:
                    return coulE + ljE + ljdispE, aux
                else:
                    return coulE + ljE + ljdispE
            else:
                if has_aux:
                    return coulE + ljE, aux
                else:
                    return coulE + ljE

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["NonbondedForce"] = NonbondedGenerator


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
            mask_list = paramset.mask[self.name]["bcc"]
            nbcc = 0
            for nnode, node in enumerate(self.ffinfo["Forces"][self.name]["node"]):
                if node["name"] == "BondChargeCorrection":
                    mask = mask_list[nbcc]
                    self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["bcc"] = bcc_now[nbcc]
                    if mask < 0.999:
                        self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["mask"] = "true"
                    nbcc += 1

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):
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
        self.mscales_coul = mscales_coul # for qeq calculation

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
            self.ethresh = kwargs.get("ethresh", 1e-5)
            self.coeff_method = kwargs.get("PmeCoeffMethod", "openmm")
            self.fourier_spacing = kwargs.get("PmeSpacing", 0.1)
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
                    charges,
                    isPBC=ifPBC,
                    topology_matrix=top_mat if self._use_bcc else None)
            if nonbondedMethod is app.NoCutoff:
                # use NoCutoff
                coulforce = CoulNoCutoffForce(
                    charges, topology_matrix=top_mat if self._use_bcc else None)
        else:
            coulforce = CoulombPMEForce(
                r_cut,
                charges, 
                kappa, (K1, K2, K3),
                topology_matrix=top_mat if self._use_bcc else None)

        coulenergy = coulforce.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux=None):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)

            if self._use_bcc:
                coulE = coulenergy(positions, box, pairs,
                                   params["CoulombForce"]["bcc"], mscales_coul)
            else:
                coulE = coulenergy(positions, box, pairs,
                                   mscales_coul)

            if has_aux:
                return coulE, aux
            else:
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
        paramset.addParameter(
            sig_prms, "sigma", field=self.name, mask=sig_mask)
        paramset.addParameter(eps_prms, "epsilon",
                              field=self.name, mask=eps_mask)
        paramset.addParameter(sig_nbfix, "sigma_nbfix",
                              field=self.name, mask=sig_nbf_mask)
        paramset.addParameter(eps_nbfix, "epsilon_nbfix",
                              field=self.name, mask=eps_nbf_mask)

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
                        nonbondedCutoff, **kwargs):
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

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux=None):

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

            if has_aux:
                return ljE, aux
            else:
                return ljE

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["LennardJonesForce"] = LennardJonesGenerator


class Custom1_5BondGenerator:
    """
    A class for generating harmonic bond force field parameters.

    Attributes:
    -----------
    name : str
        The name of the force field.
    ffinfo : dict
        The force field information.
    key_type : str
        The type of the key.
    bond_keys : list of tuple
        The keys of the bonds.
    bond_params : list of tuple
        The parameters of the bonds.
    bond_mask : list of float
        The mask of the bonds.
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        """
        Initializes the HarmonicBondGenerator.

        Parameters:
        -----------
        ffinfo : dict
            The force field information.
        paramset : ParamSet
            The parameter set.
        """
        self.name = "Custom1_5BondForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        bond_keys, bond_params = [], []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            attribs = node["attrib"]
            if self.key_type is None:
                if "atomIndex1" in attribs:
                    self.key_type = "atomIndex"
                else:
                    raise ValueError(
                        "Cannot find key type for Custom1_5BondForce.")
            key = (attribs[self.key_type + "1"], attribs[self.key_type + "2"])
            bond_keys.append(key)
            k = float(attribs["k"])
            r0 = float(attribs["length"])
            bond_params.append([k, r0])

        self.bond_keys = bond_keys
        bond_length = jnp.array([i[1] for i in bond_params])
        bond_k = jnp.array([i[0] for i in bond_params])

        # register parameters to ParamSet
        paramset.addParameter(bond_length, "length",
                              field=self.name)
        # register parameters to ParamSet
        paramset.addParameter(bond_k, "k", field=self.name)

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
        """
        Overwrites the parameter set.

        Parameters:
        -----------
        paramset : ParamSet
            The parameter set.
        """
        bond_node_indices = [i for i in range(len(
            self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Bond"]

        bond_length = paramset[self.name]["length"]
        bond_k = paramset[self.name]["k"]
        for nnode, key in enumerate(self.bond_keys):
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]]["attrib"] = {
            }
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}1"] = key[0]
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"][f"{self.key_type}2"] = key[1]
            r0 = bond_length[nnode]
            k = bond_k[nnode]
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"]["k"] = str(k)
            self.ffinfo["Forces"][self.name]["node"][bond_node_indices[nnode]
                                                     ]["attrib"]["length"] = str(r0)

    def _find_key_index(self, key: Tuple[str, str]) -> int:
        """
        Finds the index of the key.

        Parameters:
        -----------
        key : tuple of str
            The key.

        Returns:
        --------
        int
            The index of the key.
        """
        for i, k in enumerate(self.bond_keys):
            if k[0] == key[0] and k[1] == key[1]:
                return i
            if k[0] == key[1] and k[1] == key[0]:
                return i
        return None

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod,
                        nonbondedCutoff, **kwargs):
        """
        Creates the potential.

        Parameters:
        -----------
        topdata : DMFFTopology
            The topology data.
        nonbondedMethod : str
            The nonbonded method.
        nonbondedCutoff : float
            The nonbonded cutoff.
        args : list
            The arguments.

        Returns:
        --------
        function
            The potential function.
        """
        bond_a1, bond_a2, bond_indices = [], [], []
        for i, k in enumerate(self.bond_keys):
            bond_a1.append(int(k[0]))
            bond_a2.append(int(k[1]))
            bond_indices.append(int(i))
        bond_a1 = jnp.array(bond_a1)
        bond_a2 = jnp.array(bond_a2)
        bond_indices = jnp.array(bond_indices)

        # 创建势函数
        harmonic_bond_force = HarmonicBondJaxForce(
            bond_a1, bond_a2, bond_indices)
        harmonic_bond_energy = harmonic_bond_force.generate_get_energy()

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet, aux=None):
            isinstance_jnp(positions, box, params)
            energy = harmonic_bond_energy(
                positions, box, pairs, params[self.name]["k"], params[self.name]["length"])
            if has_aux:
                return energy, aux
            else:
                return energy

        self._jaxPotential = potential_fn
        return potential_fn


# register the generator
_DMFFGenerators["Custom1_5BondForce"] = Custom1_5BondGenerator


class CustomGBGenerator:
    """
    A class for generating Custom Generalized Born implicit solvation models.
    The following code implements the OBC variant of the GB/SA solvation model, using the ACE approximation to estimate surface area.

    Attributes:
    -----------
    name : str
        The name of the force field.
    ffinfo : dict
        The force field information.
    key_type : str
        The type of the key.
    perParticleKey : list of tuple
        The keys of the atoms

    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        """
        Initialize the CustomGBForceGenerator

        Parameters:
        -----------
        ffinfo : dict
            The force field information.
        paramset : ParamSet
            The parameter set.

        """
        self.name = "CustomGBForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None
        self.perParticleParamIndices = [i for i in range(len(self.ffinfo["Forces"][self.name]["node"])) if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Atom"]

        perParticleKey, perParticleParam, chargeMask = [], [], []
        for i in self.perParticleParamIndices:
            attribs = self.ffinfo["Forces"][self.name]["node"][i]["attrib"]
            if self.key_type is None:
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                else:
                    raise ValueError(
                        "Cannot find key type for CustomGBForce."
                    )
            key = (attribs[self.key_type])
            perParticleKey.append(key)

            charge = float(attribs["charge"])
            radius = float(attribs["radius"])
            scale = float(attribs["scale"])

            # Parameter Charge is not trainable
            chargeMask.append(0.0)
            perParticleParam.append([charge, radius, scale])

        self.perParticleKey = perParticleKey
        paramset.addParameter(jnp.array([i[0] for i in perParticleParam]),
                              "charge", field=self.name, mask=chargeMask)
        paramset.addParameter(jnp.array([i[1] for i in perParticleParam]),
                              "radius", field=self.name)
        paramset.addParameter(jnp.array([i[2] for i in perParticleParam]),
                              "scale", field=self.name)


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
        """
        Overwrites the parameter set.

        Parameters:
        -----------
        paramset : ParamSet
            The parameter set.
        """
        radius = paramset[self.name]["radius"]
        scale = paramset[self.name]["scale"]
        for j, i in enumerate(self.perParticleParamIndices):
            self.ffinfo["Forces"][self.name]["node"][i]["attrib"]["radius"] = str(radius[j])
            self.ffinfo["Forces"][self.name]["node"][i]["attrib"]["scale"] = str(scale[j])

    def _find_key_index(self, key: Tuple[str]) -> int:
        """
        Finds the index of the key.

        Parameters:
        -----------
        key : tuple of str
            The key.

        Returns:
        --------
        int
            The index of the key.
        """
        for i, k in enumerate(self.perParticleKey):
            if k == key:
                return i
        return None

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        """
        Creates the potential.

        Parameters:
        -----------
        topdata : DMFFTopology
            The topology data.
        nonbondedMethod : str
            The nonbonded method.
        nonbondedCutoff : float
            The nonbonded cutoff.
        args : list
            The arguments.

        Returns:
        --------
        function
            The potential function.
        """
        # Load CustomGBForce parameters
        charge_indices, radius_indices, scale_indices = [], [] ,[]
        for atom in topdata.atoms():
            if self.key_type == "type":
                key = (atom.meta["type"])
            elif self.key_type == "class":
                key = (atom.meta["class"])
            idx = self._find_key_index(key)
            if idx is None:
                continue
            charge_indices.append(idx)
            radius_indices.append(idx)
            scale_indices.append(idx)

        charge_indices = jnp.array(charge_indices)
        radius_indices = jnp.array(radius_indices)
        scale_indices = jnp.array(scale_indices)

        customGBforce = CustomGBForce(charge_indices, radius_indices, scale_indices)
        GBSAOBCenergy = customGBforce.generate_get_energy()

        def potential_fn(positions: jnp.ndarray, box: jnp.ndarray, pairs: jnp.ndarray, params: ParamSet):
            pairs = pairs[:int(positions.shape[0]*(positions.shape[0]-1)/2)]
            tt = np.vstack((pairs, pairs[:,[1, 0, 2]]))
            Ipair = []
            for i in range(positions.shape[0]):
                Ipair.append([pair[1] for pair in tt if pair[0] == i])
            Ipair = jnp.array(Ipair)
            energy =  GBSAOBCenergy(positions, box, pairs, Ipair,
                           params[self.name]["charge"],
                           params[self.name]["radius"],
                           params[self.name]['scale'])
            return energy

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["CustomGBForce"] = CustomGBGenerator
