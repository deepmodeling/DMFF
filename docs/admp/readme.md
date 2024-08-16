# ADMP

Automatic Differentiable Multipolar Polarizable (ADMP) force field calculator. 

This module provides an auto-differentiable implementation of multipolar polarizable force fields, that resembles the behavior of [MPID](https://github.com/andysim/MPIDOpenMMPlugin) plugin of OpenMM. Supposedly, this module is developed for the following purposes:

1. Achieving an easy calculation of force and virial of the multipolar polarizable forcefield. 
2. Allowing fluctuating (geometric-dependent) multipoles/polarizabilities in multipolar polarizable potentials.
3. Allowing the calculation of derivatives of various force field parameters, thus achieving a more systematic and automatic parameter optimization scheme.

The module is based on [JAX](https://github.com/google/jax) and [JAX-MD](https://github.com/google/jax-md) projects. 


## Settings

In `admp/settings.py`, you can modify some global settings, including:

**PRECISION**: single or double precision

**DO_JIT**: whether do jit or not.

In admp/pme.py, you can also modify the `DEFAULT_THOLE_WIDTH` variable 
(You can directly set `dmff.admp.pme.DEFAULT_THOLE_WIDTH` in your code)


## Example

We provide a polarizable 1024 water box example in the water_fullpol folder:

```bash
cd ./examples/water_fullpol
./run.py
```

if `DO_JIT = True`, then the first run would be a bit slow, since it tries to do the jit compilation. Further executions of `get_forces` or `get_energy` should be much faster.

Following this example, we will introduce the frontend and the backend of the ADMP module.

## Frontend

In the `forcefield.xml` file, you can find the frontends of two force field components:

### ADMPDispForce

The corresponding XML node is like:

```xml
 <ADMPDispForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="1.00" mScale16="1.00">                        
   <Atom type="380" A="1203470.743" B="37.81265679" Q="-0.741706" C6="0.001383816" C8="7.27065e-05" C10="1.8076465e-6"/>
   <Atom type="381" A="83.2283563" B="37.78544799"  Q="0.370853" C6="5.7929e-05" C8="1.416624e-06" C10="2.26525e-08"/>  
 </ADMPDispForce>                                                                                                       
```

It computes the following function:

$$
\displaylines{
E = \sum_{i<j} {A_{ij}\exp(-B_{ij}r) + \left(f_1(B_{ij}, r) - 1\right)\frac{q_i q_j}{r} -f_6(B_{ij}, r)\frac{C^6_{ij}}{r^6} - \frac{C^8_{ij}}{r^8} - \frac{C^{10}_{ij}}{r^{10}}} \\
A_{ij} = \sqrt{A_i A_j} \\
B_{ij} = \sqrt{B_i B_j} \\
C_{ij}^6 = \sqrt{C_i^6 C_j^6} \\
C_{ij}^8 = \sqrt{C_i^8 C_j^8} \\
C_{ij}^{10} = \sqrt{C_i^{10} C_j^{10}} \\
f_n(r, B) = 1 - e^{-B r}\sum_{k=0}^n {\frac{(B r)^k}{k!}}
}
$$

This is actually composed by two calculators: a long-range PME calculator to tackle the C6-10 dispersion interactions, and a short-range pairwise calculator to 
compute the short-range damping and the exponential repulsion. If you only needs the long-range part, you should use `ADMPDispPmeForce`.

We do support other forms of short-range damping and repulsive potentials, which is documented in the [ADMP Frontend](frontend.md) page in details. 

The `mScales1x` attributes specify the nonbonding scaling factors applied on atom pairs that are topologically connected.

All input parameters are in `nm` and `kJ/mol`.

### ADMPPmeForce

It compute the polarizable PME force.

The corresponding XML node is like:
```xml
<ADMPPmeForce lmax="2"                                                                                                         
     mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="1.00" mScale16="1.00"                                            
     pScale12="0.00" pScale13="0.00" pScale14="0.00" pScale15="1.00" pScale16="1.00"                                            
     dScale12="1.00" dScale13="1.00" dScale14="1.00" dScale15="1.00" dScale16="1.00">                                           
   <Atom type="380" kz="381" kx="-381"                                                                                          
             c0="-0.803721"                                                                                                     
             dX="0.0" dY="0.0"  dZ="-0.00784325"                                                                                
             qXX="0.000366476" qXY="0.0" qYY="-0.000381799" qXZ="0.0" qYZ="0.0" qZZ="1.53231e-05"                               
             />                                                                                                                 
   <Atom type="381" kz="380" kx="381"                                                                                           
             c0="0.401876"                                                                                                      
             dX="-0.00121713" dY="0.0"  dZ="-0.00095895"                                                                        
             qXX="6.7161e-06" qXY="0.0" qYY="-3.37874e-05" qXZ="1.25905e-05" qYZ="0.0" qZZ="2.70713e-05"                        
             />                                                                                                                 
   <Polarize type="380" polarizabilityXX="1.1249e-03" polarizabilityYY="1.1249e-03" polarizabilityZZ="1.1249e-03" thole="0.33"/>
   <Polarize type="381" polarizabilityXX="2.6906e-04" polarizabilityYY="2.6906e-04" polarizabilityZZ="2.6906e-04" thole="0.33"/>
 </ADMPPmeForce>                                                                                                                
</ForceField>                                                                                                                   
```

In here, the `mScales`, `pScales`, and `dScales` are explained in the [theory](../user_guide/theory.md) page. `lmax` specifies
the highest order of multipoles (`lmax=2` means up to quadrupole). All input parameters that follows are in `nm` and `kJ/mol`.
Currently, we only support the isotropic polarizabilities, meaning the `XX`, `YY`, and `ZZ` components will be averaged.

The multipole moments are specified in local frames with Cartesian representations. One needs to be careful about the convention
of the Cartesian tensor: in the frontend of ADMP, we adopt the openmm convention, the quadrupole value of which is 3 times smaller 
than the convention adopted in Anthony's book.

In `run.py`, when creating the potential function, several key parameters are noted:
```python
potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
```
* `ethresh`: the (empirical) average relative error in the PME forces, which is used to setup the size of the meshgrid of PME (see [openmm doc](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald) for more details).
* `rc`: the cutoff distance. `rc`, `ethresh`, and `box` together, determine the $K_{max}$ and $\kappa$
   (please see [theory](../user_guide/theory.md)).
   Note that the `rc` variable in here is only used to determine the PME settings. The user has to make sure the `rc`
   value used in here is the same as the one used in neighbor list construction.
* `nonbondedMethod`: Currently two methods are supported: `CutoffPeriodic` and `PME` (default). When `CutoffPeriodic`
   is used, PME is turned off by setting $\kappa=0$ and removing the reciprocal space contribution.


## Backend

The backend of ADMP can be invoked independently without the frontend API, which is much more flexible. 
It can be utilized to implement fancy force fields with fluctuating atomic parameters.
***Note that all backend functions are assuming `angstrom` unit, instead of `nm`.*** This is different to the frontend!

The examples for backend are: examples/water_1024 and examples/water_pol_1024

There are mainly three calculators implemented in ADMP backend:

### ADMPPmeForce

The `ADMPPmeForce` is initialized as:

```python
pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax, lpol=True, lpme=True, steps_pol=None)
```

The inputs are:

* `box`: the 3*3 box matrix. Vectors are arranged in rows, in angstrom.

* `axis_type`: the axis type for the local frame definition for each atom:
```python
ZThenX = 0            
Bisector = 1          
ZBisect = 2           
ThreeFold = 3         
ZOnly = 4
NoAxisType = 5        
LastAxisTypeIndex = 6 
```

* `axis_indices`: indices for local axis definitions, see introduction to local frame in [theory](../user_guide/theory.md)

* `covalent_map`: a $N\times N$ ($N$ being the number of atoms) matrix of covalent spacings between atoms. 
  $n$ means the two atoms are $n$ bonds away, and 0 means the two atoms are considered as not bonded topologically.

* `rc`: cutoff distance in real space. Only used to determine the PME settings.

* `ethresh`: (empirical) average relative error in the PME forces, which is used to setup the size of the meshgrid of PME (see [openmm doc](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald) for more details).

* `lmax`: max L for multipoles.

* `lpol`: whether turn on polarization?

* `lpme`: wether turn on PME?

* `steps_pol`: specifies number of SCF steps when solving induced dipoles. If set to None, then the SCF iteration will stop 
  when the field is less than `dmff.admp.settings.POL_CONV`. Otherwise, exact `steps_pol` number of iterations will be performed.
  ***This tag is important, and needs to be set if you want to jit the function externally.***

Once the `pme_force` is initialized, the differentiable calculator can be called as:

```python
# if lpol = False
E = pme_force.get_energy(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=U)
# if lpol = True
E = pme_force.get_energy(positions, box, pairs, Q_local, mScales)
```

Important inputs include:

* `Q_local`: spherical harmonic multipole moments in local frame (in Angstrom).

* `pol`: polarizability for each atom

* `tholes`: thole width for each atom

* `mScales`, `pScales`, `dScales`: topological scaling factors

* `U_init`: initial values for induced dipoles. 


### ADMPDispPmeForce

This force computes the long range dispersion interactions with C6, C8, and C10 terms.
It is initialized and called as:

```python
disp_force = ADMPDispPmeForce(box, covalent_map, rc, ethresh, pmax, lpme=True)
E = disp_force.get_energy(positions, box, pairs, c_list, mScales)
```

Most parameters are similar to the PME force, several other important parameters include:

* `pmax`: maximum power of dispersion, usually 10.

* `c_list`: the ***square root*** of dispersion coefficients. (N*3) array, that is,
  sqrt(C6), sqrt(C8), and sqrt(C10) of each atom.


### Short-Range Pairwise Force

DMFF provides a simple approach to raise a distance-dependent pair interaction kernel into
a many-body potential function. Suppose you want to implement a function looks like this:

$$
E = \sum_{ij} {f(r_{ij}; a_i, a_j, b_i, b_j, c_i, c_j)}
$$

Where $a,b,c$ are atomic parameters.

Then you need to define a pairwise kernel looks like:

```python
from jax import vmap, jit

def f_kernel(r, m, a_i, a_j, b_i, b_j, c_i, c_j):
    # do calculations, get E
    return E * m

f_kernel = vmap(jit(f_kernel))
```

Then raise it to a many-body potential for a particular system:

```python
potential_fn = generate_pairwise_interaction(f_kernel, covalent_map)
# a_list, b_list, c_list are atomic parameter lists
E = potential_fn(positions, box, pairs, mScales, a_list, b_list, c_list)
```

The resulted `potential_fn` function will take care the following business for you:

1. Topological scaling based on `covalent_map` and `mScales`;

2. Assign parameters for each interacting pair;

We can use this to immplement short-range repulsion and short-range damping quite easily.





