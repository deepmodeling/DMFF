import numpy as np
from scipy import constants

j2ev = constants.physical_constants["joule-electron volt relationship"][0]
# kJ/mol to eV/particle
energy_coeff = j2ev * constants.kilo / constants.Avogadro
# vacuum electric permittivity in eV^-1 * angstrom^-1
epsilon = constants.epsilon_0 / constants.elementary_charge * constants.angstrom
# qqrd2e = 1 / (4 * np.pi * epsilon)
# # eV
# dielectric = 1 / (4 * np.pi * epsilon)
# kJ/mol
DIELECTRIC = 1 / (4 * np.pi * epsilon) / energy_coeff
# DIELECTRIC = 1389.35455846
SQRT_PI = np.sqrt(np.pi)

__all__ = ["DIELECTRIC", "SQRT_PI"]
