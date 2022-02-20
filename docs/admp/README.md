# ADMP

Automatic Differentiable Multipolar Polarizable (ADMP) force field calculator. 

This module provides an auto-differentiable implementation of multipolar polarizable force fields, that resembles the behavior of [MPID](https://github.com/andysim/MPIDOpenMMPlugin) plugin of OpenMM. Supposedly, this module is developed for the following purposes:

1. Achieving an easy calculation of force and virial of the multipolar polarizable forcefield. 
2. Allowing fluctuating (geometric-dependent) multipoles/polarizabilities in multipolar polarizable potentials.
3. Allowing the calculation of derivatives of various force field parameters, thus achieving a more systematic and automatic parameter optimization scheme.

The module is based on [JAX](https://github.com/google/jax) and [JAX-MD](https://github.com/google/jax-md) projects. 



## Installation

### Dependencies

ADMP module depends on the following packages, install them before using ADMP:

1. Install [jax](https://github.com/google/jax) (pick the correct cuda version, see more details on their installation guide):

   ```bash
   pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

2. Install [jax-md](https://github.com/google/jax-md) :

   ```bash
   pip install jax-md --upgrade
   ```

   ADMP currently relies on the space and partition modules to provide neighbor list

3. Install ADMP:

   ADMP is a pure python module, just simply put it in your $PYTHONPATH.

   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/admp	
   ```



## Settings

In `admp/settings.py`, you can modify some global settings, including:

**PRECISION**: single or double precision

**DO_JIT**: whether do jit or not.



## Example

We provide a MPID 1024 water box example. In water_1024 and water_pol_1024, we show both the nonpolarizable and the polarizable cases.

```bash
cd ./examples/water_1024
./run_admp.py

cd ./examples/water_pol_1024
./run_admp.py
```

if `DO_JIT = True`, then the first run would be a bit slow, since it tries to do the jit compilation. Further executions of `get_forces` or `get_energy` should be much faster.

