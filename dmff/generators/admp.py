from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException, isinstance_jnp
from ..admp.pme import setup_ewald_parameters
import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from ..admp.disp_pme import ADMPDispPmeForce
from ..admp.multipole import convert_cart2harm, convert_harm2cart
from ..admp.pairwise import (
    TT_damping_qq_c6_kernel,
    generate_pairwise_interaction,
    slater_disp_damping_kernel,
    slater_sr_kernel, ## no Hard Core Potential
    slater_sr_hc_kernel,  ## added Hard Core Potential
    TT_damping_qq_kernel,
)
from ..admp.pme import ADMPPmeForce


class ADMPDispGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "ADMPDispForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        # default params
        self._jaxPotential = None
        self.types = []
        self.ethresh = 5e-4
        self.pmax = 10

        default_scales = [0.0, 0.0, 0.0, 1.0, 1.0]

        mScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"mScale1{i}", default_scales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        mScales.append(1.0)
        self.mScales = jnp.array(mScales)

        A, B, Q, C6, C8, C10 = [], [], [], [], [], []
        self.atom_keys = []
        atom_mask = []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                self.atom_keys.append(attribs[self.key_type])
                A.append(float(attribs["A"]))
                B.append(float(attribs["B"]))
                Q.append(float(attribs["Q"]))
                C6.append(float(attribs["C6"]))
                C8.append(float(attribs["C8"]))
                C10.append(float(attribs["C10"]))
                mask = 1.0
                if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                    mask = 0.0
                atom_mask.append(mask)
        A = jnp.array(A)
        B = jnp.array(B)
        Q = jnp.array(Q)
        C6 = jnp.array(C6)
        C8 = jnp.array(C8)
        C10 = jnp.array(C10)
        self.atom_keys = np.array(self.atom_keys)
        atom_mask = jnp.array(atom_mask)
        paramset.addParameter(A, "A", field=self.name, mask=atom_mask)
        paramset.addParameter(B, "B", field=self.name, mask=atom_mask)
        paramset.addParameter(Q, "Q", field=self.name, mask=atom_mask)
        paramset.addParameter(C6, "C6", field=self.name, mask=atom_mask)
        paramset.addParameter(C8, "C8", field=self.name, mask=atom_mask)
        paramset.addParameter(C10, "C10", field=self.name, mask=atom_mask)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        atom_mask = paramset.mask[self.name]["A"]
        A = paramset[self.name]["A"]
        B = paramset[self.name]["B"]
        Q = paramset[self.name]["Q"]
        C6 = paramset[self.name]["C6"]
        C8 = paramset[self.name]["C8"]
        C10 = paramset[self.name]["C10"]

        node2atom = [
            i
            for i in range(len(self.ffinfo["Forces"][self.name]["node"]))
            if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Atom"
        ]

        for natom in range(len(self.atom_keys)):
            nnode = node2atom[natom]
            A_new = A[natom]
            B_new = B[natom]
            Q_new = Q[natom]
            C6_new = C6[natom]
            C8_new = C8[natom]
            C10_new = C10[natom]
            mask = atom_mask[natom]
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["A"] = str(A_new)
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["B"] = str(B_new)
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["Q"] = str(Q_new)
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C6"] = str(
                C6_new
            )
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C8"] = str(
                C8_new
            )
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C10"] = str(
                C10_new
            )
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"][
                    "mask"
                ] = "true"

    def _find_atype_key_index(self, atype: str):
        for n, i in enumerate(self.atom_keys):
            if i == atype:
                return n
        return None

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPDispForce")
        if nonbondedMethod is app.CutoffPeriodic:
            lpme = False
        else:
            lpme = True

        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = atoms[i].meta[self.key_type]
            map_atomtype[i] = self._find_atype_key_index(atype)
        topdata._meta[self.name+"_map_atomtype"] = map_atomtype
        # here box is only used to setup ewald parameters, no need to be differentiable
        if lpme:
            box = topdata.getPeriodicBoxVectors() * 10
        else:
            box = jnp.ones((3, 3))
        # get the admp calculator
        if unit.is_quantity(nonbondedCutoff):
            rc = nonbondedCutoff.value_in_unit(unit.angstrom)
        else:
            rc = nonbondedCutoff * 10.0

        # get calculator
        if "ethresh" in kwargs:
            self.ethresh = kwargs["ethresh"]

        Force_DispPME = ADMPDispPmeForce(box, rc, self.ethresh, self.pmax, lpme=lpme)
        self.disp_pme_force = Force_DispPME
        pot_fn_lr = Force_DispPME.get_energy
        pot_fn_sr = generate_pairwise_interaction(
            TT_damping_qq_c6_kernel, static_args={}
        )

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux=None):
            # Assume nm for frontend functions, still uses Angstrom for backend
            positions = positions * 10
            box = box * 10

            params = params[self.name]
            mScales = self.mScales
            a_list = (
                params["A"][map_atomtype] / 2625.5
            )  # kj/mol to au, as expected by TT_damping kernel
            b_list = params["B"][map_atomtype] * 0.0529177249  # nm^-1 to au
            q_list = params["Q"][map_atomtype]
            # the input parameters are assumed to be in nm too, need to convert to Angstrom
            c6_list = jnp.sqrt(params["C6"][map_atomtype] * 1e6)
            c8_list = jnp.sqrt(params["C8"][map_atomtype] * 1e8)
            c10_list = jnp.sqrt(params["C10"][map_atomtype] * 1e10)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))

            E_sr = pot_fn_sr(
                positions, box, pairs, mScales, a_list, b_list, q_list, c_list[0]
            )
            E_lr = pot_fn_lr(positions, box, pairs, c_list.T, mScales)
            if has_aux:
                return E_sr - E_lr, aux
            else:
                return E_sr - E_lr

        self._jaxPotential = potential_fn
        return potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


_DMFFGenerators["ADMPDispForce"] = ADMPDispGenerator


class ADMPDispPmeGenerator:
    r"""
    This one computes the undamped C6/C8/C10 interactions
    u = \sum_{ij} c6/r^6 + c8/r^8 + c10/r^10
    """

    def __init__(self, ffinfo, paramset):
        self.name = "ADMPDispPmeForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        self._jaxPotential = None
        self.atom_types = None
        self.ethresh = 5e-4
        self.pmax = 10
        self.name = "ADMPDispPmeForce"
        self._meta = {}
        default_scales = [0.0, 0.0, 0.0, 1.0, 1.0]

        mScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"mScale1{i}", default_scales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        mScales.append(1.0)
        self.mScales = jnp.array(mScales)

        C6, C8, C10 = [], [], []
        self.atom_keys = []
        atom_mask = []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                self.atom_keys.append(attribs[self.key_type])
                C6.append(float(attribs["C6"]))
                C8.append(float(attribs["C8"]))
                C10.append(float(attribs["C10"]))
                mask = 1.0
                if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                    mask = 0.0
                atom_mask.append(mask)
        C6 = jnp.array(C6)
        C8 = jnp.array(C8)
        C10 = jnp.array(C10)
        atom_mask = jnp.array(atom_mask)
        self.atom_keys = np.array(self.atom_keys)
        paramset.addParameter(C6, "C6", field=self.name, mask=atom_mask)
        paramset.addParameter(C8, "C8", field=self.name, mask=atom_mask)
        paramset.addParameter(C10, "C10", field=self.name, mask=atom_mask)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        atom_mask = paramset.mask[self.name]["sigma"]
        C6 = paramset[self.name]["C6"]
        C8 = paramset[self.name]["C8"]
        C10 = paramset[self.name]["C10"]

        node2atom = [
            i
            for i in range(len(self.ffinfo["Forces"][self.name]["node"]))
            if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Atom"
        ]

        for natom in range(len(self.atom_keys)):
            nnode = node2atom[natom]
            C6_new = C6[natom]
            C8_new = C8[natom]
            C10_new = C10[natom]
            mask = atom_mask[natom]
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C6"] = str(
                C6_new
            )
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C8"] = str(
                C8_new
            )
            self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C10"] = str(
                C10_new
            )
            if mask < 0.999:
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"][
                    "mask"
                ] = "true"

    def _find_atype_key_index(self, atype: str):
        for n, i in enumerate(self.atom_keys):
            if i == atype:
                return n
        return None

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPDispPmeForce")
        if nonbondedMethod is app.CutoffPeriodic:
            lpme = False
        else:
            lpme = True

        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = atoms[i].meta[self.key_type]
            map_atomtype[i] = self._find_atype_key_index(atype)

        topdata._meta[self.name+"_map_atomtype"] = map_atomtype

        # here box is only used to setup ewald parameters, no need to be differentiable
        if lpme:
            box = topdata.getPeriodicBoxVectors() * 10
        else:
            box = jnp.ones((3, 3))
        # get the admp calculator
        if unit.is_quantity(nonbondedCutoff):
            rc = nonbondedCutoff.value_in_unit(unit.angstrom)
        else:
            rc = nonbondedCutoff * 10.0

        # get calculator
        if "ethresh" in kwargs:
            self.ethresh = kwargs["ethresh"]

        disp_force = ADMPDispPmeForce(box, rc, self.ethresh, self.pmax, lpme)
        self.disp_force = disp_force
        pot_fn_lr = disp_force.get_energy

        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True
        else:
            has_aux = False

        def potential_fn(positions, box, pairs, params, aux = None):
            positions = positions * 10
            box = box * 10
            params = params[self.name]
            C6_list = params["C6"][map_atomtype] * 1e6  # to kj/mol * A**6
            C8_list = params["C8"][map_atomtype] * 1e8
            C10_list = params["C10"][map_atomtype] * 1e10
            c6_list = jnp.sqrt(C6_list)
            c8_list = jnp.sqrt(C8_list)
            c10_list = jnp.sqrt(C10_list)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))
            E_lr = pot_fn_lr(positions, box, pairs, c_list.T, self.mScales)
            if has_aux:
                return -E_lr, aux
            else:
                return -E_lr

        self._jaxPotential = potential_fn
        # self._top_data = data
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

    def getMetaData(self):
        return self._meta


_DMFFGenerators["ADMPDispPmeForce"] = ADMPDispPmeGenerator


class QqTtDampingGenerator:
    r"""
    This one calculates the tang-tonnies damping of charge-charge interaction
    E = \sum_ij exp(-B*r)*(1+B*r)*q_i*q_j/r
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.ffinfo = ffinfo
        self._jaxPotnetial = None
        self.name = "QqTtDampingForce"
        paramset.addField(self.name)
        self.key_type = None
        self.atom_keys = []

        # get mscales
        default_scales = [0.0, 0.0, 0.0, 1.0, 1.0]

        self.mScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"mScale1{i}", default_scales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        self.mScales.append(1.0)
        self.mScales = jnp.array(self.mScales)

        # get atomtypes
        B, Q = [], []
        atom_mask = []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                self.atom_keys.append(attribs[self.key_type])
                B.append(float(attribs["B"]))
                Q.append(float(attribs["Q"]))
                mask = 1.0
                if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                    mask = 0.0
                atom_mask.append(mask)
        B = jnp.array(B)
        Q = jnp.array(Q)
        atom_mask = jnp.array(atom_mask)
        self.atom_keys = np.array(self.atom_keys)
        paramset.addParameter(B, "B", field=self.name, mask=atom_mask)
        paramset.addParameter(Q, "Q", field=self.name, mask=atom_mask)

    def getName(self) -> str:
        return self.name

    def overwrite(self, paramset):
        B = paramset[self.name]["B"]
        Q = paramset[self.name]["Q"]
        atom_mask = paramset.mask[self.name]["B"]

        nnode = 0
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                B_new = B[nnode]
                Q_new = Q[nnode]
                mask = atom_mask[nnode]
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["B"] = str(
                    B_new
                )
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["Q"] = str(
                    Q_new
                )
                if mask < 0.999:
                    self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"][
                        "mask"
                    ] = "true"
                nnode += 1

    def _find_atype_key_index(self, atype: str):
        for n, i in enumerate(self.atom_keys):
            if i == atype:
                return n
        return None

    # on working
    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = atoms[i].meta[self.key_type]
            map_atomtype[i] = np.where(self.atom_keys == atype)[0][0]

        topdata._meta[self.name+"_map_atomtype"] = map_atomtype
        pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_kernel, static_args={})

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux = None):
            positions = positions * 10
            box = box * 10
            params = params[self.name]
            b_list = params["B"][map_atomtype] / 10  # convert to A^-1
            q_list = params["Q"][map_atomtype]

            E_sr = pot_fn_sr(positions, box, pairs, self.mScales, b_list, q_list)
            if has_aux:
                return E_sr, aux
            else:
                return E_sr

        self._jaxPotential = potential_fn
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential



# register all parsers
_DMFFGenerators["QqTtDampingForce"] = QqTtDampingGenerator


class SlaterDampingGenerator:
    r"""
    This one computes the slater-type damping function for c6/c8/c10 dispersion
    E = \sum_ij (f6-1)*c6/r6 + (f8-1)*c8/r8 + (f10-1)*c10/r10
    fn = f_tt(x, n)
    x = br - (2*br2 + 3*br) / (br2 + 3*br + 3)
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "SlaterDampingForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self._jaxPotential = None

        # get mscales
        default_scales = [0.0, 0.0, 0.0, 1.0, 1.0]
        self.mScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"mScale1{i}", default_scales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        self.mScales.append(1.0)
        self.mScales = jnp.array(self.mScales)

        # get atomtypes
        self.atom_keys = []
        B, C6, C8, C10 = [], [], [], []
        atom_mask = []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                self.atom_keys.append(attribs[self.key_type])
                B.append(float(attribs["B"]))
                C6.append(float(attribs["C6"]))
                C8.append(float(attribs["C8"]))
                C10.append(float(attribs["C10"]))
                mask = 1.0
                if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                    mask = 0.0
                atom_mask.append(mask)
        B = jnp.array(B)
        C6 = jnp.array(C6)
        C8 = jnp.array(C8)
        C10 = jnp.array(C10)
        atom_mask = jnp.array(atom_mask)
        self.atom_keys = np.array(self.atom_keys)
        paramset.addParameter(B, "B", field=self.name, mask=atom_mask)
        paramset.addParameter(C6, "C6", field=self.name, mask=atom_mask)
        paramset.addParameter(C8, "C8", field=self.name, mask=atom_mask)
        paramset.addParameter(C10, "C10", field=self.name, mask=atom_mask)

    def getName(self) -> str:
        return self.name

    def overwrite(self, paramset):
        B = paramset[self.name]["B"]
        C6 = paramset[self.name]["C6"]
        C8 = paramset[self.name]["C8"]
        C10 = paramset[self.name]["C10"]
        atom_mask = paramset.mask[self.name]["B"]

        nnode = 0
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                B_new = B[nnode]
                C6_new = C6[nnode]
                C8_new = C8[nnode]
                C10_new = C10[nnode]
                mask = atom_mask[nnode]
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["B"] = B_new
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C6"] = C6_new
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C8"] = C8_new
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["C10"] = C10_new
                if mask < 0.999:
                    self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"][
                        "mask"
                    ] = "true"
                nnode += 1

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = atoms[i].meta[self.key_type]
            map_atomtype[i] = np.where(self.atom_keys == atype)[0][0]

        topdata._meta[self.name+"_map_atomtype"] = map_atomtype

        # WORKING
        pot_fn_sr = generate_pairwise_interaction(
            slater_disp_damping_kernel, static_args={}
        )

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux={}):
            positions = positions * 10
            box = box * 10
            params = params[self.name]
            b_list = params["B"][map_atomtype] / 10  # convert to A^-1
            c6_list = jnp.sqrt(params["C6"][map_atomtype] * 1e6)  # to kj/mol * A**6
            c8_list = jnp.sqrt(params["C8"][map_atomtype] * 1e8)
            c10_list = jnp.sqrt(params["C10"][map_atomtype] * 1e10)
            E_sr = pot_fn_sr(
                positions,
                box,
                pairs,
                self.mScales,
                b_list,
                c6_list,
                c8_list,
                c10_list,
            )
            if has_aux:
                return E_sr, aux
            else:
                return E_sr

        self._jaxPotential = potential_fn
        # self._top_data = data
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


_DMFFGenerators["SlaterDampingForce"] = SlaterDampingGenerator


class SlaterExGenerator:
    r"""
    This one computes the Slater-ISA type exchange interaction
    u = \sum_ij A * (1/3*(Br)^2 + Br + 1)
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet, default_name=None):
        if default_name is None:
            self.name = "SlaterExForce"
        else:
            self.name = default_name
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self._jaxPotential = None
        self.key_type = None

        # get mscales
        default_scales = [0.0, 0.0, 0.0, 1.0, 1.0]
        self.mScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"mScale1{i}", default_scales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        self.mScales.append(1.0)
        self.mScales = jnp.array(self.mScales)

        # get atomtypes
        self.atom_keys = []
        A, B = [], []
        atom_mask = []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                if "type" in attribs:
                    self.key_type = "type"
                elif "class" in attribs:
                    self.key_type = "class"
                self.atom_keys.append(attribs[self.key_type])
                A.append(float(attribs["A"]))
                B.append(float(attribs["B"]))
                mask = 1.0
                if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                    mask = 0.0
                atom_mask.append(mask)
        A = jnp.array(A)
        B = jnp.array(B)
        atom_mask = jnp.array(atom_mask)
        self.atom_keys = np.array(self.atom_keys)
        paramset.addParameter(A, "A", field=self.name, mask=atom_mask)
        paramset.addParameter(B, "B", field=self.name, mask=atom_mask)

    def getName(self) -> str:
        return self.name

    def overwrite(self, paramset):
        A = paramset[self.name]["A"]
        B = paramset[self.name]["B"]
        atom_mask = paramset.mask[self.name]["B"]

        nnode = 0
        for node in self.ffinfo["Forces"][self.name]["node"]:
            if node["name"] == "Atom":
                attribs = node["attrib"]
                A_new = A[nnode]
                B_new = B[nnode]
                mask = atom_mask[nnode]
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["A"] = A_new
                self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"]["B"] = B_new
                if mask < 0.999:
                    self.ffinfo["Forces"][self.name]["node"][nnode]["attrib"][
                        "mask"
                    ] = "true"
                nnode += 1

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = atoms[i].meta[self.key_type]
            map_atomtype[i] = np.where(self.atom_keys == atype)[0][0]

        topdata._meta[self.name+"_map_atomtype"] = map_atomtype

        pot_fn_sr = generate_pairwise_interaction(slater_sr_hc_kernel, static_args={}) 
        #slater_ex_sr_kernel: added Hard Core Potential

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux=None): 
            positions = positions * 10
            box = box * 10
            params = params[self.name]
            a_list = params["A"][map_atomtype]
            b_list = params["B"][map_atomtype] / 10  # nm^-1 to A^-1

            energy = pot_fn_sr(positions, box, pairs, self.mScales, a_list, b_list) 
            if has_aux:
                return energy, aux
            else:
                return energy

        self._jaxPotential = potential_fn
        # self._top_data = data
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential



_DMFFGenerators["SlaterExForce"] = SlaterExGenerator



# Here are all the short range "charge penetration" terms
# They all have the exchange form with minus sign
class SlaterSrEsGenerator(SlaterExGenerator):
    def __init__(self, ffinfo: dict, paramset: ParamSet, default_name=None):
        if default_name is None:
            super().__init__(ffinfo, paramset, default_name="SlaterSrEsForce")
        else:
            super().__init__(ffinfo, paramset, default_name=default_name)
    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = atoms[i].meta[self.key_type]
            map_atomtype[i] = np.where(self.atom_keys == atype)[0][0]

        topdata._meta[self.name+"_map_atomtype"] = map_atomtype

        pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel, static_args={}) 
        ## slater_sr_others_kernel: no Hard Core Potential 

        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True

        def potential_fn(positions, box, pairs, params, aux=None): 
            positions = positions * 10
            box = box * 10
            params = params[self.name]
            a_list = params["A"][map_atomtype]
            b_list = params["B"][map_atomtype] / 10  # nm^-1 to A^-1
            energy = - pot_fn_sr(positions, box, pairs, self.mScales, a_list, b_list)

            # add minus sign
            if has_aux:
                return energy, aux
            else:
                return energy

        self._jaxPotential = potential_fn
        # self._top_data = data
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


class SlaterSrPolGenerator(SlaterSrEsGenerator):
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        super().__init__(ffinfo, paramset, default_name="SlaterSrPolForce")


class SlaterSrDispGenerator(SlaterSrEsGenerator):
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        super().__init__(ffinfo, paramset, default_name="SlaterSrDispForce")


class SlaterDhfGenerator(SlaterSrEsGenerator):
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        super().__init__(ffinfo, paramset, default_name="SlaterDhfForce")


# register all parsers
_DMFFGenerators["SlaterSrEsForce"] = SlaterSrEsGenerator
_DMFFGenerators["SlaterSrPolForce"] = SlaterSrPolGenerator
_DMFFGenerators["SlaterSrDispForce"] = SlaterSrDispGenerator
_DMFFGenerators["SlaterDhfForce"] = SlaterDhfGenerator


class ADMPPmeGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "ADMPPmeForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)

        # default params
        self._jaxPotential = None
        self.multipole_types = []
        self.polarize_types = []
        self.ethresh = 5e-4
        self.step_pol = None
        self.ref_dip = ""
        self.pme_force = None

        self.lmax = int(self.ffinfo["Forces"][self.name]["meta"]["lmax"])

        default_mscales = [0.0, 0.0, 1.0, 1.0, 1.0]
        default_pscales = [0.0, 0.0, 1.0, 1.0, 1.0]
        default_dscales = [1.0, 1.0, 1.0, 1.0, 1.0]

        mScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"mScale1{i}", default_mscales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        pScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"pScale1{i}", default_pscales[i - 2]
                )
            )
            for i in range(2, 7)
        ]
        dScales = [
            float(
                self.ffinfo["Forces"][self.name]["meta"].get(
                    f"dScale1{i}", default_dscales[i - 2]
                )
            )
            for i in range(2, 7)
        ]

        # make sure the last digit is 1.0
        mScales.append(1.0)
        pScales.append(1.0)
        dScales.append(1.0)
        self.mScales = jnp.array(mScales)
        self.pScales = jnp.array(pScales)
        self.dScales = jnp.array(dScales)

        # check if polarize
        polarize_nodes = [
            node
            for node in self.ffinfo["Forces"][self.name]["node"]
            if node["name"] == "Polarize"
        ]
        if len(polarize_nodes) > 0:
            self.lpol = True
        else:
            self.lpol = False

        # get atom types
        multipole_nodes = [
            node
            for node in self.ffinfo["Forces"][self.name]["node"]
            if node["name"] in ["Multipole", "Atom"]
        ]
        c0, dX, dY, dZ, qXX, qYY, qZZ, qXY, qXZ, qYZ = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        kxs, kys, kzs = [], [], []
        multipole_masks = []
        for nnode, node in enumerate(multipole_nodes):
            attribs = node["attrib"]
            if "type" in attribs:
                self.key_type = "type"
            elif "class" in attribs:
                self.key_type = "class"
            self.multipole_types.append(attribs[self.key_type])
            # record local coords as kz kx ky
            # None if not occur
            kx = attribs.get("kx", "")
            ky = attribs.get("ky", "")
            kz = attribs.get("kz", "")
            kxs.append(kx)
            kys.append(ky)
            kzs.append(kz)
            # record multipoles
            c0.append(float(attribs["c0"]))
            if self.lmax >= 1:
                dX.append(float(attribs["dX"]))
                dY.append(float(attribs["dY"]))
                dZ.append(float(attribs["dZ"]))
            else:
                dX.append(0.0)
                dY.append(0.0)
                dZ.append(0.0)
            if self.lmax >= 2:
                qXX.append(float(attribs["qXX"]))
                qYY.append(float(attribs["qYY"]))
                qZZ.append(float(attribs["qZZ"]))
                qXY.append(float(attribs["qXY"]))
                qXZ.append(float(attribs["qXZ"]))
                qYZ.append(float(attribs["qYZ"]))
            else:
                qXX.append(0.0)
                qYY.append(0.0)
                qZZ.append(0.0)
                qXY.append(0.0)
                qXZ.append(0.0)
                qYZ.append(0.0)
            mask = 1.0
            if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                mask = 0.0
            multipole_masks.append(mask)
        multipole_masks = jnp.array(multipole_masks)

        # invoke by `self.kStrings["kz"][itype]`
        self.kStrings = {}
        self.kStrings["kx"] = kxs
        self.kStrings["ky"] = kys
        self.kStrings["kz"] = kzs

        # assume that polarize tag match the per atom type
        # pol_XX = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityXX')
        # pol_YY = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityYY')
        # pol_ZZ = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityZZ')
        # thole_0 = self.fftree.get_attribs(f'{self.name}/Polarize', 'thole')
        if self.lpol:
            pol = []
            thole = []
            polarizability_masks = []
            for nnode, node in enumerate(polarize_nodes):
                self.polarize_types.append(node["attrib"][self.key_type])
                polarizabilityXX = float(node["attrib"]["polarizabilityXX"])
                polarizabilityYY = float(node["attrib"]["polarizabilityYY"])
                polarizabilityZZ = float(node["attrib"]["polarizabilityZZ"])
                pol.append(
                    (polarizabilityXX + polarizabilityYY + polarizabilityZZ) / 3.0
                )
                thole.append(float(node["attrib"]["thole"]))
                mask = 1.0
                if (
                    "mask" in node["attrib"]
                    and node["attrib"]["mask"].upper() == "TRUE"
                ):
                    mask = 0.0
                polarizability_masks.append(mask)
            polarizability_masks = jnp.array(polarizability_masks)
            pol = jnp.array(pol) * 1000.0
            paramset.addParameter(
                pol, "pol", field=self.name, mask=polarizability_masks
            )
            paramset.addParameter(
                jnp.array(thole), "thole", field=self.name, mask=polarizability_masks
            )

        n_atoms = len(self.multipole_types)

        # assert n_atoms == len(polarizabilityXX), "Number of polarizabilityXX does not match number of atoms!"

        # map atom multipole moments
        if self.lmax == 0:
            n_mtps = 1
        elif self.lmax == 1:
            n_mtps = 4
        elif self.lmax == 2:
            n_mtps = 10
        Q = np.zeros((n_atoms, n_mtps))

        # TDDO: unit conversion
        Q[:, 0] = c0
        if self.lmax >= 1:
            Q[:, 1] = dX
            Q[:, 2] = dY
            Q[:, 3] = dZ
            Q[:, 1:4] *= 10
        if self.lmax >= 2:
            Q[:, 4] = qXX
            Q[:, 5] = qYY
            Q[:, 6] = qZZ
            Q[:, 7] = qXY
            Q[:, 8] = qXZ
            Q[:, 9] = qYZ
            Q[:, 4:10] *= 300

        # add all differentiable params to self.params
        Q_local = convert_cart2harm(jnp.array(Q), self.lmax)
        paramset.addParameter(Q_local, "Q_local", field=self.name, mask=multipole_masks)

    def getName(self):
        return self.name

    def overwrite(self, paramset):
        Q_global = convert_harm2cart(paramset[self.name]["Q_local"], self.lmax)
        q_local_masks = paramset.mask[self.name]["Q_local"]
        polar_masks = paramset.mask[self.name]["pol"]

        n_multipole, n_pol = 0, 0
        for nnode in range(len(self.ffinfo["Forces"][self.name]["node"])):
            node = self.ffinfo["Forces"][self.name]["node"][nnode]
            if node["name"] in ["Atom", "Multipole"]:
                node["attrib"]["c0"] = Q_global[n_multipole, 0]
                if self.lmax >= 1:
                    node["attrib"]["dX"] = Q_global[n_multipole, 1] * 0.1
                    node["attrib"]["dY"] = Q_global[n_multipole, 2] * 0.1
                    node["attrib"]["dZ"] = Q_global[n_multipole, 3] * 0.1
                if self.lmax >= 2:
                    node["attrib"]["qXX"] = Q_global[n_multipole, 4] / 300.0
                    node["attrib"]["qYY"] = Q_global[n_multipole, 5] / 300.0
                    node["attrib"]["qZZ"] = Q_global[n_multipole, 6] / 300.0
                    node["attrib"]["qXY"] = Q_global[n_multipole, 7] / 300.0
                    node["attrib"]["qXZ"] = Q_global[n_multipole, 8] / 300.0
                    node["attrib"]["qYZ"] = Q_global[n_multipole, 9] / 300.0
                if q_local_masks[n_multipole] < 0.999:
                    node["mask"] = "true"
                n_multipole += 1
            elif node["name"] == "Polarize":
                node["attrib"]["polarizabilityXX"] = paramset[self.name]["pol"][n_pol] * 0.001
                node["attrib"]["polarizabilityYY"] = paramset[self.name]["pol"][n_pol] * 0.001
                node["attrib"]["polarizabilityZZ"] = paramset[self.name]["pol"][n_pol] * 0.001
                node["attrib"]["thole"] = paramset[self.name]["thole"][n_pol]
                if polar_masks[n_pol] < 0.999:
                    node["attrib"]["mask"] = "true"
                n_pol += 1

    def _find_multipole_key_index(self, atype: str):
        for n, i in enumerate(self.multipole_types):
            if i == atype:
                return n
        return None

    def _find_polarize_key_index(self, atype: str):
        for n, i in enumerate(self.polarize_types):
            if i == atype:
                return n
        return None
    
    @staticmethod
    def setAxisType(kIndices):
        # setting up axis_indices and axis_type
        ZThenX = 0
        Bisector = 1
        ZBisect = 2
        ThreeFold = 3
        ZOnly = 4  # typo fix
        NoAxisType = 5
        LastAxisTypeIndex = 6

        # set axis type

        ky = kIndices[2]
        kyNegative = False
        if ky.startswith('-'):
            ky = ky[1:]
            kyNegative = True

        kx = kIndices[1]
        kxNegative = False
        if kx.startswith('-'):
            kx = kx[1:]
            kxNegative = True

        kz = kIndices[0]
        kzNegative = False
        if kz.startswith('-'):
            kz = kz[1:]
            kzNegative = True

        axisType = ZThenX
        if (not kz):
            axisType = NoAxisType
        if (kz and not kx):
            axisType = ZOnly
        if (kz and kzNegative or kx and kxNegative):
            axisType = Bisector
        if (kx and kxNegative and ky and kyNegative):
            axisType = ZBisect
        if (kz and kzNegative and kx and kxNegative and ky and kyNegative):
            axisType = ThreeFold

        return axisType, [kz, kx, ky]

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.CutoffNonPeriodic: "CutoffNonPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPPmeForce")
        if nonbondedMethod in [app.CutoffPeriodic, app.NoCutoff, app.CutoffNonPeriodic]:
            lpme = False
        else:
            lpme = True

        n_atoms = topdata.getNumAtoms()
        map_atomtype = np.zeros(n_atoms, dtype=int)
        map_poltype = np.zeros(n_atoms, dtype=int)

        atoms = [a for a in topdata.atoms()]
        for i in range(n_atoms):
            atype = atoms[i].meta[
                self.key_type
            ]  # convert str to int to match atomTypes
            map_atomtype[i] = self._find_multipole_key_index(atype)
            if self.lpol:
                map_poltype[i] = self._find_polarize_key_index(atype)


        # here box is only used to setup ewald parameters, no need to be differentiable
        box = topdata.getPeriodicBoxVectors()
        if box is not None:
            box = jnp.array(box) * 10.0
        else:
            box = jnp.eye(3)
        # get the admp calculator
        if unit.is_quantity(nonbondedCutoff):
            rc = nonbondedCutoff.value_in_unit(unit.angstrom)
        else:
            rc = nonbondedCutoff * 10.0

        # build covalent map
        covalent_map = topdata.buildCovMat()
        # build intra-molecule axis
        # the following code is the direct transplant of forcefield.py in openmm 7.4.0

        if self.lmax > 0:
            # setting up axis_indices and axis_type
            ZThenX = 0
            Bisector = 1
            ZBisect = 2
            ThreeFold = 3
            ZOnly = 4  # typo fix
            NoAxisType = 5
            LastAxisTypeIndex = 6

            axis_types = {}
            axis_indices = {}
            patched_atoms = np.zeros(n_atoms, dtype=int)
            for i_type_rev in range(len(self.multipole_types)):
                i_type = len(self.multipole_types) - i_type_rev - 1
                t = self.multipole_types[i_type]  # reverse
                # find all the atoms patched the multipole type
                i_atoms = [
                    i for i in range(n_atoms) if atoms[i].meta[self.key_type] == t
                ]
                for i_atom in i_atoms:
                    if patched_atoms[i_atom] > 0:
                        continue

                    kz = self.kStrings["kz"][i_type]
                    kx = self.kStrings["kx"][i_type]
                    ky = self.kStrings["ky"][i_type]
                    axisType, (kz, kx, ky) = self.setAxisType([kz, kx, ky])

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1
                    # try to assign multipole parameters
                    neighbors = np.where(covalent_map[i_atom] == 1)[0]
                    if len(neighbors) < 1 and axisType != NoAxisType:
                        continue
                    if axisType == ZThenX:
                        if kz is None or kx is None:
                            raise DMFFException("ZThenX axis requires both kz and kx!")
                        # find zaxis
                        for i_neighbor in neighbors:
                            if kz == atoms[i_neighbor].meta[self.key_type]:
                                zaxis = i_neighbor
                                break
                        if zaxis < 0:
                            continue
                        # find xaxis on 1-2 pairs
                        for i_neighbor in neighbors:
                            if i_neighbor == zaxis:
                                continue
                            if kx == atoms[i_neighbor].meta[self.key_type]:
                                xaxis = i_neighbor
                                break
                        if xaxis < 0:
                            # find xaxis on 1-3 pairs
                            neighbors2 = np.where(covalent_map[zaxis] == 1)[0]
                            for j_neighbor in neighbors2:
                                if j_neighbor == i_atom:
                                    continue
                                if kx == atoms[j_neighbor].meta[self.key_type]:
                                    xaxis = j_neighbor
                                    break
                        if xaxis < 0:
                            continue
                    elif axisType == ZOnly:
                        # 1-2 only
                        for i_neighbor in neighbors:
                            if kz == atoms[i_neighbor].meta[self.key_type]:
                                zaxis = i_neighbor
                                break
                        if zaxis < 0:
                            continue
                    elif axisType == Bisector:
                        for i_neighbor in neighbors:
                            if kz == atoms[i_neighbor].meta[self.key_type]:
                                zaxis = i_neighbor
                                break
                        if zaxis < 0:
                            continue
                        for j_neighbor in neighbors:
                            if j_neighbor == zaxis:
                                continue
                            if kx == atoms[j_neighbor].meta[self.key_type]:
                                xaxis = j_neighbor
                                break
                        if xaxis < 0:
                            continue
                    elif axisType in [ZBisect, ThreeFold]:
                        for i_neighbor in neighbors:
                            if kz == atoms[i_neighbor].meta[self.key_type]:
                                zaxis = i_neighbor
                                break
                        if zaxis < 0:
                            continue
                        for j_neighbor in neighbors:
                            if j_neighbor == zaxis:
                                continue
                            if kx == atoms[j_neighbor].meta[self.key_type]:
                                xaxis = j_neighbor
                                break
                        if xaxis < 0:
                            continue
                        for k_neighbor in neighbors:
                            if k_neighbor == zaxis or k_neighbor == xaxis:
                                continue
                            if ky == atoms[k_neighbor].meta[self.key_type]:
                                yaxis = k_neighbor
                                break
                        if yaxis < 0:
                            continue
                    elif axisType == NoAxisType:
                        pass
                    else:
                        continue

                    axis_types[i_atom] = axisType
                    axis_indices[i_atom] = [zaxis, xaxis, yaxis]
                    patched_atoms[i_atom] = 1

            for i_atom in range(patched_atoms.shape[0]):
                if patched_atoms[i_atom] < 0.5:
                    raise DMFFException("Atom %d not matched in forcefield!" % i_atom)

            axis_indices = np.array([axis_indices[i] for i in range(n_atoms)])
            axis_types = np.array([axis_types[i] for i in range(n_atoms)])
        else:
            axis_types = np.zeros(n_atoms, dtype=int)
            axis_indices = None

        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True
        else:
            has_aux = False

        if "ethresh" in kwargs:
            self.ethresh = kwargs["ethresh"]
        if "step_pol" in kwargs:
            self.step_pol = kwargs["step_pol"]
        pme_force = ADMPPmeForce(
            box,
            map_atomtype, 
            axis_types,
            axis_indices,
            rc,
            self.ethresh,
            self.lmax,
            self.lpol,
            lpme,
            self.step_pol,
            has_aux
        )
        self.pme_force = pme_force
        topdata._meta[self.name+"_map_atomtype"] = map_atomtype
        topdata._meta[self.name+"_map_poltype"] = map_poltype

        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True
        else:
            has_aux = False

        def potential_fn(positions, box, pairs, params, aux=None):
            positions = positions * 10.0
            box = box * 10.0
            Q_local = params["ADMPPmeForce"]["Q_local"][map_atomtype]
            if self.lpol:
                pol = params["ADMPPmeForce"]["pol"][map_poltype]
                tholes = params["ADMPPmeForce"]["thole"][map_poltype]

                if has_aux:
                    if aux is not None:
                        energy, aux = pme_force.get_energy(
                            positions, box, pairs, Q_local, pol, tholes,
                            self.mScales, self.pScales, self.dScales,
                            U_init = aux["U_ind"], aux = aux
                        )
                    else:
                        energy, aux = pme_force.get_energy(
                            positions, box, pairs, Q_local, pol, tholes, 
                            self.mScales, self.pScales, self.dScales, 
                            U_init=jnp.zeros((n_atoms,3)), aux={}
                        )
                    return energy, aux
                else:
                    energy = pme_force.get_energy(
                        positions, box, pairs, Q_local, pol, tholes,
                        self.mScales, self.pScales, self.dScales,
                        U_init = pme_force.U_ind
                    )
                    return energy
            else:
                energy = pme_force.get_energy(
                            positions, box, pairs, Q_local, self.mScales
                        )
                if has_aux:
                    return energy, aux
                else:
                    return energy

        self._jaxPotential = potential_fn
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


_DMFFGenerators["ADMPPmeForce"] = ADMPPmeGenerator
