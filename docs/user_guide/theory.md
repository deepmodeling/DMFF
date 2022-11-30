# 5. Theory

DMFF project aims to implement organic molecular force fields using a differentiable programming framework, such that derivatives with respect to atomic positions, box shape, and force field parameters can be easily computed. It contains different modules, dealing with different types of force field terms. Currently, there are two primary modules:

1. ADMP (**A**utomatic **D**ifferentiable **M**ultipolar **P**olarizable Potential) module
  ADMP mainly deals with multipolar polarizable models. Its core function is very similar to the MPID plugin in OpenMM, implementing PME calculators for multipolar polarizable electrostatic interactions and long-range dispersion interactions (with the shape of $c_i c_j/r^p$). It also devises a user-defined real-space pairwise interaction calculator based on cutoff scheme.
  
2. Classical module
  The classical module implements conventional (AMBER and OPLS like) force fields. For long-range interactions, it invokes the ADMP PME kernel, but wrapps it in a more "classical" way. It also incoporates the classical intramolecular terms: bonds, angles, proper and improper dihedrals etc.

All interations involved in DMFF are briefly introduced below and the users are encouraged to read the references for more mathematical details:

## 5.1 Electrostatic Interaction

The electrostatic interaction between two atoms can be described using multipole expansion, in which the electron cloud of an atom can be expanded as a series of multipole moments including charges, dipoles, quadrupoles, and octupoles etc. If only the charges (zero-moment) are considered, it is reduced to the point charge model in classical force fields:

$$
V=\sum_{ij} \frac{q_i q_j}{r_{ij}}
$$

where $q_i$ is the charge of atom $i$.

More complex (and supposedly more accurate) force field can be obtained by including more multipoles with higher orders. Some force fields, such as MPID, goes as high as octupoles. Currently in DMFF, we support up to quadrupoles:

$$
V=\sum_{tu} Q_t^A T^{AB}_{tu} Q_u^B
$$

where $Q_t^A$ represents the t-component of the multipole moment of atom A. Note there are two (equivalent) ways to define multipole moments: cartesian and spherical harmonics. Cartesian representation is over-complete but with a simpler definition, while spherical harmonics are easier to use in real calculations. In the user API, we use cartesian representation, in consistent with the AMOEBA and the MPID plugins in OpenMM. However, spherical harmonics are always used in the computation kernel, and we assume all components are arranged in the following order:

$$0, 10, 1c, 1s, 20, 21c, 21s, 22c, 22s, ...$$

The $T_{tu}^{AB}$ represents the interaction tensor between multipoles. The mathematical expression of these tensors can be found in the appendix F of Ref 1. The user can also find the conversion rule between different representations in Ref 1 & 5.


## 5.2 Coordinate System for Multipoles

Different to charges, the definition of multipole moments depends on the coordinate system. The exact value of the moment tensor will be rotated in accord to different coordinate systems. There are three types of frames involved in DMFF, each used in a different scenario:

  - Global frame: coordinate system binds to the simulation box. It is same for all the atoms. We use this frame to calculate the charge density structure factor $S(\vec{k})$ in reciprocal space.
  - Local frame: this frame is defined differently on each atom, determined by the positions of its peripheral atoms. Normally, atomic multipole moments are most stable in the local frame, so it is the most suitable frame for force field input. In DMFF API, the local frames are defined using the same way as the AMOEBA plugin in OpenMM. The details can found in the following references:
      * [OpenMM forcefield.py](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py#L4894), line 4894~4933
      * [J. Chem. Theory Comput. 2013, 9, 9, 4046–4063](https://pubs.acs.org/doi/abs/10.1021/ct4003702)
  - Quasi internal frame, aka. QI frame: this frame is defined for each pair of interaction sites, in which the z-axis is pointing from one site to another. In this frame, the real-space interaction tensor ($T_{tu}^{AB}$) can be greatly simplified due to symmetry. We thus use this frame in the real space calculation of PME.


## 5.3 Polarization Interaction

DMFF supports polarizable force fields, in which the dipole moment of the atom can respond to the change of the external electric field. In practice, each atom has not only permanent multipoles $Q_t$, but also induced dipoles $U_{ind}$. The induced dipole-induced dipole and induced dipole-permanent multipole interactions needs to be damped at short-range to avoid polarization catastrophe. In DMFF, we use the Thole damping scheme identical to MPID (ref 6), which introduces a damping width ($a_i$) for each atom $i$. The damping function is then computed and applied to the corresponding interaction tensor. Taking $U_{ind}$-permanent charge interaction as an example, the definition of damping function is:

$$
\displaylines{
1-\left(1+a u+\frac{1}{2} a^{2} u^{2}\right) e^{-a u} \\ 
a=a_i + a_j \\ 
u=r_{ij}/\left(\alpha_i \alpha_j\right)^{1/6} 
}
$$

Other damping functions between multipole moments can be found in Ref 6, table I. 

It is noted that the atomic damping parameter $a=a_i+a_j$ is only effective on topological neighboring pairs (with $pscale = 0$), while a default value of $a_{default}$ is set for all other pairs. In DMFF, the atomic $a_i$ is specified via the xml API, while $a_{default}$  is controlled by the `dmff.admp.pme.DEFAULT_THOLE_WIDTH` variable, which is set to 5.0 by default.

We solve $U_{ind}$ by minimizing the electrostatic energy:

$$
V=V_{perm-perm}+V_{perm-ind}+V_{ind-ind}
$$

The last two terms are related to $U_{ind}$. Without introducing the nonlinear polarization terms (e.g., some force fields introduce $U^4$ to avoid polarization catastrophe), the last two terms are quadratic to $U_{ind}$: 

$$
V_{perm-ind}+V_{ind-ind}=U^TKU-FU
$$

where the off-diagonal term of $K$ matrix is induced-induced dipole interaction, the diagonal term is formation energy of the induced dipoles ($\sum_i \frac{U_i^2}{2\alpha_i}$); the $F$ matrix represents permanent multipole - induced dipole interaction. We use the gradient descent method to optimize energy to get $U_{ind}$.

In the current version, we temporarily assume that the polarizability is spherically symmetric, thus the polarizability $\alpha_i$ is a scalar, not a tensor. **Thus the inputs (`polarizabilityXX, polarizabilityYY, polarizabilityZZ`) in the xml API is averaged internally**. In future, it is relatively simple to relax this restriction: simply change the reciprocal of the polarizability to the inverse of the matrix when calculating the diagonal terms of the $K$ matrix.

## 5.4 Dispersion Interaction

In ADMP, we assume that the following expansion is used for the long-range dispersion interaction:

$$
V_{disp}=\sum_{ij}-\frac{C_{ij}^6}{r_{ij}^6}-\frac{C_{ij}^8}{r_{ij}^8}-\frac{C_{ij}^{10}}{r_{ij}^{10}}-...
$$

where the dispersion coefficients are determined by the following combination rule:

$$
C^n_{ij}=\sqrt{C_i^n C_j^n}
$$

Note that the dispersion terms should be consecutive even powers according to the perturbation theory, so the odd dispersion terms are not supported in ADMP. 

In ADMP, this long-range dispersion is computed using PME (*vida infra*), just as electrostatic terms.

In the classical module, dispersions are treated as short-range interactions using standard cutoff scheme.

## 5.5 Long-Range Interaction with PME

The long-range potential includes electrostatic, polarization, and dispersion (in ADMP) interactions. Taking charge-charge interaction as example, the interaction decays in the form of $O(\frac{1}{r})$, and its energy does not converge with the increase of cutoff distance. The multipole electrostatics and dispersion interactions also converge slow with respect to cutoff distance. We therefore use Particle Meshed Ewald(PME) method to calculate these interactions.

In PME, the interaction tensor is splitted into the short-range part and the long-range part, which are tackled in real space and reciprocal space, respectively. For example, the Coulomb interaction is decomposed as:

$$
\frac{1}{r}=\frac{erfc(\kappa r)}{r}+\frac{erf(\kappa r)}{r}
$$

The first term is a short-range term, which can be calculated directly by using a simple distance cutoff in real space. The second term is a long-range term, which needs to be calculated in reciprocal space by fast Fourier transform(FFT). The total energy of charge-charge interaction is computed as:

$$
\displaylines{
E_{real} = \sum_{ij}\frac{erfc(\kappa r_{ij})}{r_{ij}}  \\
E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}\exp\left[\frac{k^2}{4\kappa^2}\right]\left|S(\vec{k})\right|^2}\frac{1}{\left|\theta(\vec{k})\right|^2} \\ 
E_{self} = -\frac{\kappa}{\sqrt{\pi}}\sum_i {q_i^2} \\
E = E_{real}+E_{recip}+E_{self}
}
$$

As for multipolar PME and dispersion PME, the users and developers are referred to Ref 2, 3, and 5 for mathematical details.

The key parameters in PME include:

  - $\kappa$: controls the separation of the long-range and the short-range. The larger $\kappa$ is, the faster the real space energy decays, the smaller the cutoff distance can be used in the real space, and more difficult it is to converge the reciprocal energy and the larger $K_{max}$ it needs;

  - $r_{c}$: cutoff distance in real space;

  - $K_{max}$: controls the number of maximum k-points in all three dimensions


In DMFF, we determine these parameters in the same way as in [OpenMM](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald):

$$
\displaylines{
\kappa=\sqrt{-\log (2 \delta)} / r_{c} \\ 
K_{max}=\frac{2 \kappa d}{3 d^{1 / 5}}
}
$$

where the user needs to specify the cutoff distance $r_c$ when building the neighbor list, the width of the box in each dimension $d$ (determined from the input box matrix), and the energy accuracy $\delta$.

In the current version, the dispersion PME calculator uses the same parameters as in electrostatic PME.

## 5.6 Short-Range Interaction

Short-range pair interaction refers to all interactions with the following form:

$$
V=\sum_{ij}v(r_{ij})
$$

Some common short-range pair interactions include:

  -  The repulsive part of the Buckingham or the Lennard-Jones potential:

$$
\displaylines{
v(r)=A\exp(-\beta r) \\ 
v(r)=\frac{C^{12}}{r^{12}}
}
$$

  - Tang-Tonnies Damping: damping function for short-range electrostatic and dispersion energies.

$$
f_n(r,\beta)=1-e^{-\beta r} \sum_{k=0}^{n}\frac{(\beta r)^k}{k!}
$$

In ADMP, the user can define a pairwise kernel function $f(dr)=f(dr, m, a_i,a_j,b_i,b_j,\dots)$ ($a_i, b_i$ are atomic parameters), then use `generate_pairwise_interaction` to raise the kernel function into an energy calculator (see details in ADMP manual).

## 5.7 Combination Rule

For most traditional force fields, pairwise parameters between interacting particles are determined by atomic parameters. This mathematical relationship is called the combination rule. For example, in the calculation of LJ potential, the following combination rule may be used:    

$$
\displaylines{
v(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right] \\ 
\sigma=\frac{\sigma_i + \sigma_j}{2} \\ 
\varepsilon=\sqrt{\varepsilon_i \varepsilon_j}
}
$$

In ADMP module, we do not make any assumptions about the specific mathematical forms of the combination rule and $v(r)$. Users need to write them in the definition of the pairwise kernel function.

## 5.8 Neighbor List

All DMFF real space calculations depends on neighbor list (or "pair list" as we sometimes call in DMFF). Its purpose is to keep a record of all the "neighbors" within a certain distance of the central atom, thus avoiding to go over all pairs explicitly.

In DMFF, we use external code ([jax-md](https://github.com/google/jax-md)) to build such neighbor list. An input argument named `pairs` is required in all real-space calculators, which contains the indices of all "interacting pairs" (i.e., pairs within a certain distance $r_c$). We assume that the `pairs` variable is in `ordered sparse` format in Jax-md. That is, a $N_p\times2$ index array with $N_p$ being the number of interacting pairs. It is noted that change of $N_p$ leads to recompilation of the code and significantly slows down the calculation. Therefore, jax-md usually buffers the list such that $N_p$ remains a constant in the simulation. DMFF expects the buffer part of the neighbor list is filled with $N+1$, with $N$ being the total number of atoms in the system. 

Since the pair list only provides atom **id** information, it does not take part in the differentiation process, so it can be fed in as a normal numpy array (instead of a jax numpy array).

## 5.9 Topological scaling

In order to avoid double-counting with the bonding term, we often need to scale the non-bonding interactions between two atoms that are topologically connected. The scaling factor depends on the topological distance between the two atoms. We define two atoms separated by one bond as "1-2" interaction, and those separated by two bonds as "1-3" interaction, and so on. For example, in the OPLS-AA force field, all "1-2" nonbonding interactions are turned off completely, while all "1-3" non-bonding interactions are scaled by 50%. DMFF supports such feature, and important variables related to topological scaling include:

  - `covalent_Map`: a $N\times N$ matrix, which defines the topological spacings between atoms. If the matrix element is 0, it indicates that the **topological** distance between the two atoms is too far (or the two atoms are not connected), so the nonbonding interaction is fully turned on between them.

  - `mScales`: The list of scaling factors. The first element is the scaling factor for all 1-2 nonbonding interaction, the second element is the scaling for 1-3 interactions, and so on. The list can be of any length, but the last number of the list **must be 1**, which represents the complete, unscaled nonbonding interaction.

  - `pScales`/`dScales`: similar to `mScales`, but only related to polarizable calculations. They are scaling factors for induced-perm and induced-induced interactions, respectively. 


## 5.10 General Many-Body Interactions

(such as ML force field) TBA

## 5.11 Bonded Interaction

Intramolecular bonding interactions refer to all interactions that depend on internal coordinates (IC), such as bonds, angles, and dihedrals, etc.

+ Harmonic Bonding Terms
  
    The definition of the bonding term in DMFF is the same as in OpenMM. For each bond, we have:

$$
E=\frac{1}{2}k(x-x_0)^2
$$
  
+ Harmonic Angle Terms

$$
E=\frac{1}{2} k\left(\theta-\theta_{0}\right)^{2}
$$
  
+ Dihedral Terms
  
    1. Proper dihedral
    2. Improper dihedral
  
+ Multi IC coupling term

## 5.12 Typification

Before energy calculation, atomic and IC parameters (such as charge, multipole moment, dispersion coefficient, polarizability, force constant of each bond and angle, etc.) need to be assigned first. 

Generally, these parameters should be dependent on the chemical and geometric environment of each atom and IC. However, in  conventional force field, in order to reduce the number of parameters, atoms and ICs are classified according to their topological environment, and atoms/ICS in the same class would share parameters. The process of classifying each atom and IC and assigning the corresponding parameters according to their class is called typification.

In DMFF, the input parameters that need to be optimized are called **force field parameters**, and the parameters of each atom and IC after typification are called **atomic parameters**. Note that in an ideal force field, if we can directly predict atomic parameters using machine learning model, the process of typification is *not necessary*. Therefore, in DMFF, we decouple the typification code with the computation kernels, so that the core calculators based on atomic parameters has their own API and can be invoked independently. The typification code, in combination with the xml/pdb input parsers, composes the high-level API (the `dmff.api` module).

The design of the high-level DMFF API is based on the existing framework of OpenMM. DMFF needs to keep the derivation chain uninterrupted when dispatching the force field params into atomic params. Therefore, maintaining the basic design logic of OpenMM, we rewrite the typification part of OpenMM using Jax. Briefly speaking, OpenMM/DMFF requires the users to clearly define the type of each atom in each residue and the connection mode between atoms in residue templates. Then the residue templates are used to match the PDB file to typify the whole system. See the following [documents](../dev_guide/arch.MD) for details.

## 5.13 References

1. [Anthony's book](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199672394.001.0001/acprof-9780199672394)
2. [The Multipolar Ewald paper in JCTC:  J. Chem. Theory Comput. 2015, 11, 2, 436–450](https://pubs.acs.org/doi/abs/10.1021/ct5007983)
3. [The dispersion Ewald/PME](https://aip.scitation.org/doi/pdf/10.1063/1.470117)
4. [Frenkel & Smit book](https://www.elsevier.com/books/understanding-molecular-simulation/frenkel/978-0-12-267351-1)
5. Note on multipole ewald. [link](multipole_pme.md)
6. [MPID Reference](https://doi.org/10.1063/1.4984113)
