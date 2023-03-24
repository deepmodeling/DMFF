from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..utils import DMFFException, isinstance_jnp
from ..admp.pme import setup_ewald_parameters
import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from ..classical.inter import CoulNoCutoffForce, CoulombPMEForce, CoulReactionFieldForce


class CoulombGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "CoulombForce"
        self.ffinfo = ffinfo
        self.paramset = paramset
        self.coulomb14scale = float(
            self.ffinfo["Forces"]["CoulombForce"]["meta"]["coulomb14scale"])
        self._use_bcc = False

    def overwrite(self):
        # paramset to ffinfo
        pass

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
            self.coulomb14scale)

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

        if nonbondedMethod is not app.PME:
            # do not use PME
            if nonbondedMethod in [app.CutoffPeriodic, app.CutoffNonPeriodic]:
                # use Reaction Field
                coulforce = CoulReactionFieldForce(r_cut,
                                                   charges,
                                                   isPBC=ifPBC,
                                                   topology_matrix=cov_mat if self._use_bcc else None)
            if nonbondedMethod is app.NoCutoff:
                # use NoCutoff
                coulforce = CoulNoCutoffForce(
                    topology_matrix=cov_mat if self._use_bcc else None)
        else:
            coulforce = CoulombPMEForce(r_cut,
                                        kappa, (K1, K2, K3),
                                        topology_matrix=cov_mat if self._use_bcc else None)

        coulenergy = coulforce.generate_get_energy()

        def potential_fn(positions, box, pairs, params):

            # check whether args passed into potential_fn are jnp.array and differentiable
            # note this check will be optimized away by jit
            # it is jit-compatiable
            isinstance_jnp(positions, box, params)

            if self._use_bcc:
                coulE = coulenergy(positions, box, pairs,
                                   charges, params["CoulombForce"]["bcc"], mscales_coul)
            else:
                coulE = coulenergy(positions, box, pairs,
                                   charges, mscales_coul)

            return coulE

        self._jaxPotential = potential_fn
        return potential_fn
