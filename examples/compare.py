import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

var1 = jnp.load('openmm_api/pol.npy')
var2 = jnp.load('water_pol_1024/pol.npy')
npt.assert_allclose(var1, var2)

var1 = jnp.load('openmm_api/mScales.npy')
var2 = jnp.load('water_pol_1024/mScales.npy')
npt.assert_allclose(var1, var2)

var1 = jnp.load('openmm_api/dScales.npy')
var2 = jnp.load('water_pol_1024/dScales.npy')
npt.assert_allclose(var1, var2)

var1 = jnp.load('openmm_api/Q_local.npy')
var2 = jnp.load('water_pol_1024/Q_local.npy')
npt.assert_allclose(var1, var2, rtol=1e-6)

var1 = jnp.load('openmm_api/tholes.npy')
var2 = jnp.load('water_pol_1024/tholes.npy')
npt.assert_allclose(var1, var2)

var1 = jnp.load('openmm_api/U_ind.npy')
var2 = jnp.load('water_pol_1024/U_ind.npy')
npt.assert_allclose(var1, var2)