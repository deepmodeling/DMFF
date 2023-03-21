from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit


class CoulombGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "CoulombForce"
        self.ffinfo = ffinfo
        self.paramset = paramset
        self.paramset.addField(self.name)
        self.paramset.addParameter([
            self.ffinfo["Forces"]["CoulombForce"]["attrib"]["coulomb14scale"]
        ],
                                   "coulomb14scale",
                                   field=self.name)

    def overwrite(self):
        # paramset to ffinfo
        self.ffinfo["Forces"]["CoulombForce"]["attrib"][
            "coulomb14scale"] = self.paramset[self.name]["coulomb14scale"][0]

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
        mscales_coul = mscales_coul.at[2].set(
            self.paramset["coulomb14scale"][0])

        # set PBC
        if nonbondedMethod not in [app.NoCutoff, app.CutoffNonPeriodic]:
            ifPBC = True
        else:
            ifPBC = False

        charges = [a.meta["charge"] for a in topdata.atoms()]
        charges = jnp.array(charges)

        cov_mat = topdata.buildCovMat()

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

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff

        if nonbondedMethod is not app.PME:
            # do not use PME
            if nonbondedMethod in [app.CutoffPeriodic, app.CutoffNonPeriodic]:
                # use Reaction Field
                coulforce = CoulReactionFieldForce(r_cut,
                                                   map_charge,
                                                   isPBC=ifPBC,
                                                   topology_matrix=cov_mat)
            if nonbondedMethod is app.NoCutoff:
                # use NoCutoff
                coulforce = CoulNoCutoffForce(charges, topology_matrix=cov_mat)
        else:
            coulforce = CoulombPMEForce(r_cut,
                                        charges,
                                        kappa, (K1, K2, K3),
                                        topology_matrix=cov_mat)

        coulenergy = coulforce.generate_get_energy()

        def potential_fn(positions, box, pairs, params):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)

            coulE = coulenergy(positions, box, pairs,
                                charges, mscales_coul)

            return coulE

        self._jaxPotential = potential_fn