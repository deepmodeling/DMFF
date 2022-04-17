# Theory background

This project aims to implement an organic force field with a differentiable framework to automatically derivate atomic position, box shape, force field parameters, and other inputs. 

The ADMP force field module has the following interactions：


## Electrostatic term

We can use multipole expansion with cutoff for electrostatic interaction between electron could of atoms. If only the charge (zero-moment) part is retained, it reduces to the point charge model in the classical force field:

$$V=\sum_{ij} \frac{q_i q_j}{r_{ij}}$$

where $q_i$ is the charge number of the atom.

More complex force field forms can be obtained by increasing truncation of the moment order. Some force fields, such as AMOEBA and MPID, use higher-order truncation. In DMFF, we have up to four moments:

$$V=\sum_{tu} \hat{Q}_t^A T^{AB}_{tu} \hat{Q}_u^B$$

where $Q_t^A$ represents the t-component of multipole moment of atom A, there are two definitions: cartesian coordinates and spherical harmonics. In DMFF, we use spherical harmonics. The sequence is:

$$0, 10, 1c, 1s, 20, 21c, 21s, 22c, 22s, ...$$

The $T_{tu}^{AB}$ represents the interaction tensor among multipoles, which mathematical expression can refer to Ref 1 appendix F. The conversion between different multipole moment definitions and rotation rules can refer to Ref 1 & 5.

## Multipole moment coordinate system

Different from charge, multipole moment definition depends on the coordinate system. What we use are mainly three:

  - global frame: coordinate system binds to the simulation box. It same for all atoms. We use this system to calculate charge density structure factor $$S(\vec{k})$$ in reciprocal space.
  - local frame: this system defines each atom's coordinate by the positions of its peripheral atoms, and then multipole moment is given under this system. Generally, atomic multipole moments have considerable stability in the local frame, so it is more suitable as a force field parameter input. In DMFF, the definition of the local frame is the same as the AMOEBA plugin in OpenMM. The detail can refer to the following literatures:
      * [OpenMM forcefield.py](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py#L4894), line 4894~4933
      * [J. Chem. Theory Comput. 2013, 9, 9, 4046–4063](https://pubs.acs.org/doi/abs/10.1021/ct4003702)
  - quasi internal frame, aka. QI frame: a unique coordinate system to calculate the interaction between two sites in real space. Taking the connecting line of two sites as the Z-axis, the interaction tensor can be greatly simplified by using symmetry under this coordinate system to $$T_{tu}^{AB}$$.

## Polarization term

DMFF supports polarizable force fields, in which the dipole moment of the atom can respond to the change of the external electric field. We support that each atom not only has permanent multipoles $\hat{Q}_t$ but induced dipole $U_{ind}$. The interaction between induced-induced dipole and induced-permanent dipole needs damping, which mathematical expression is the same as MPID(Ref 6). Specifically, each atom needs a thole parameter ($$a_i$$). When calculating the interaction of polarizable sites, the damping function will be introduced. Take $$U_{ind}$$-permanent charge interaction as an example, the definition of damping function is:

$$1-\left(1+a u+\frac{1}{2} a^{2} u^{2}\right) e^{-a u} \\ a=a_i + a_j \\ u=r_{ij}/\left(\alpha_i \alpha_j\right)^{1/6} $$

Other damping form between multipole moment can refer to Ref 6, table I.

We solve $$U_{ind}$$ by minimizing electrostatic energy. The energy can be written as follows:

$$V=V_{perm-perm}+V_{perm-ind}+V_{ind-ind}$$

the last two-termterms relate to $U_{ind}$. Without introducing the non-linear polarization term(e.g., some force fields introduce $$U^4$$ to avoid polarization catastrophe), the last two terms is quadratic function of $$U_{ind}$$: 

$$V_{perm-ind}+V_{ind-ind}=U^TKU-FU$$

where the off-diagonal term of $K$ matrix is induced-induced dipole interaction, the diagonal term is form energy of induced dipole($\sum_i \frac{U_i^2}{2\alpha_i}$); the $F$matrix represents permanent multipole - induced dipole interaction. We use the gradient descent method to optimize energy to get $U_{ind}$.

In the current version, we temporarily assume that the polarizability is spherically symmetric; the polarizability $a_i$ is a scalar, not a tensor. It is relatively simple to relax this restriction; change the reciprocal of polarizability to the inverse of the matrix when calculating the diagonal term of the $K$ matrix.

## Dispersion term

We assume that the following expansion can describe the dispersion between atoms.

$$V_{disp}=\sum_{ij}-\frac{C_{ij}^6}{r_{ij}^6}-\frac{C_{ij}^8}{r_{ij}^8}-\frac{C_{ij}^{10}}{r_{ij}^{10}}-...$$

where dispersion factor satisfies the following combination rule:

$$C^n_{ij}=\sqrt{C_i^n C_j^n}$$

Note that the dispersion terms should be continuous even powers according to the perturbation theory, so the odd dispersion terms are not supported in DMFF.

## Long-range interaction with PME

The long-range interaction will be involved in treating electrostatic, polarization, and dispersion interactions. Take charge-charge interaction as an example. The interaction decays in the form of $O(\frac{1}{r})$, and its energy does not converge with the increase of cutoff distance. In calculating multipole and dispersion forces, the convergence speed of cutoff distance is also slow. We introduce Particle Meshed Ewald(PME) method to calculate those interactions.

An example, as charge-charge interaction, we split the interaction tensor by long and short-range in PME:

$$\frac{1}{r}=\frac{erfc(\kappa r)}{r}+\frac{erf(\kappa r)}{r}$$

The first term is a short-range term, which can be calculated directly by using a simple distance cutoff in real space. The second term is a long-range term, which needs to be calculated in reciprocal space by fast Fourier transform(FFT). The total energy of charge-charge interaction is:

$$E_{real}=\sum_{ij}\frac{erfc(\kappa r_{ij})}{r_{ij}}\\E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}\exp\left[\frac{k^2}{4\kappa^2}\right]\left|S(\vec{k})\right|^2}\frac{1}{\left|\theta(\vec{k})\right|^2}\\E_{self}=-\frac{\kappa}{\sqrt{\pi}}\sum_i {q_i^2} \\E=E_{real}+E_{recip}+E_{self}$$

As for multipole PME and dispersion PME, the expression refer to Ref 2, Ref 3, and Ref 5.

The key parameters in PME include:

  - $\kappa$: controls the separation degree of long-range and short-range. The greater $\kappa$, the faster decay of the real space, the smaller the cutoff distance that can be used in the real space, and the more difficult the convergence of the reciprocal space and the more k-points are required;

  - $ r_{c}$: cutoff distance in real space;

  - $ K_{max}$: controls the number of maximum k-points


In DMFF, we determine these parameters in the same way as the PME in [OpenMM](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald):

$$\kappa=\sqrt{-\log (2 \delta)} / r_{c} \\ K_{max}=\frac{2 \kappa d}{3 d^{1 / 5}}$$

where the user needs to specify the cutoff distance $r_c$ when building the neighbor list, the width of the box in each dimension $d$, and the energy calculation accuracy $\delta$.

In the current version, the parameter determination mechanism of multipole PME and dispersion PME is exactly the same as that of charge PME.

## Short-range interaction

Short distance pair interaction refers to all interactions with the following forms:

$V=\sum_{ij}v(r_{ij})$

Some common short-range pair interactions include:

  -  Buckingham or excluded part of LJ potential:

$$v(r)=A\exp(-\beta r)\\v(r)=\frac{C^{12}}{r^{12}}$$

  - Tang-Tonnies Damping: Damping function for short-range electrostatic and dispersive effects
    1. Combination Rule:

For most traditional force fields, pairwise parameters in each pair of interactions are determined by atomic parameters. This mathematical relationship is called the combination rule. For example, in the calculation of LJ potential, the following combination rule may be used:    

$$v(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]\\ \sigma=\frac{\sigma_i + \sigma_j}{2} \\ \varepsilon=\sqrt{\varepsilon_i \varepsilon_j}$$

DMFF does not make any assumptions about the specific mathematical forms of the combination rule and $v(r)$. Users need to write them in the definition of the pairwise kernel function.

## Neighborlist

The calculation of all real spaces depends on the construction of the neighborlist. Its purpose is to find the "nearest neighbor" within a certain distance of the central atom by using an efficient method and calculating its interaction.

In DMFF, we use external code([jax-md](https://github.com/google/jax-md)) to build the nearest neighbor list. An external input argument named pairs is required in all real-space code, which contains the sequence numbers of all atomic pairs within a certain distance $r_c$. We assume that the pairs variable is in the form of `sparse` or `ordered sparse` in Jax-md.

Because the nearest neighbor list only provides atom **id** information, it has nothing to do with the derivation chain.

## Topological scaling

In the organic molecular force field, in order to avoid double-counting with the bonding interaction term, we generally need to scale the non-bonding interaction between two atoms link to each other in topology. The specific scaling degree depends on the topological distance between the two atoms. We generally define two atoms with one bond as "1-2" interaction, and those separated by two bonds as "1-3" interaction, and so on. For example, in the OPLS-AA force field, all "1-2" non-bonding actions are turned off, while all "1-3" non-bonding actions are scaled to 50% of the normal non-bond actions.

Important variables related to topological scaling in DMFF include:

  - `covalent_Map`: the format is $N*N$ matrix, which defines the topological spacing between any two atoms. If the matrix element is 0, it indicates that the **topological** distance between the two atoms is too far, so there is a complete non-bonding interaction between them.

  - `mScales`: The sequence of scaling factors is stored. The first element is 1-2 non-bond scaling, the second element is 1-3 non-bond scaling, and so on. The last number of the sequence **must be 1**, which represents the complete, unscaled non-bond interaction.

  - `pScales`/`dScales`: It is only related to calculating polarization energy, representing induced-perm and induced-induced, respectively. The format is the same as that of `mScales`.


## General multibody terms (such as ML force field)

  TODO:

## Nonding interaction

Intramolecular bonding interactions refer to all interactions that depend on internal coordinates(aka. IC), which mainly include bond, angle, dihedral, etc.

  * Harmonic Bonding Terms
    The definition of binding term in DMFF is the same as that in OpenMM. For each key, we have:
    $$E=\frac{1}{2}k(x-x_0)^2$$
    Note prefactor $1/2$ before force constant
  
  * Harmonic Angle Terms
    we have: 
    $$E=\frac{1}{2} k\left(\theta-\theta_{0}\right)^{2}$$
  
  * Dihedral Terms
    1. Proper dihedral
    2. Improper dihedral
  
  * Multi IC coupling term

## Atomic classification

Before energy calculation, atomic and IC parameters (such as charge, multipole moment, dispersion coefficient, polarizability, force constant of each bond and angle, etc.) need to be assigned. Generally speaking, these parameters should be related to each atom's specific environment and conformation or IC. However, in the traditional force field, in order to reduce the number of parameters, atoms and ICs are generally classified according to their topological environment, and atoms in the same class or ICs share parameters. Classifying each atom and IC and assigning its parameters is called typification.

In DMFF, the input parameters that need to be optimized are called **force field parameters**, and the parameters of each atom and IC after typing are called **atomic parameters**. Note that in the new ideal force field, if we can directly predict atomic parameters using machine learning model, the process of typification is *not necessary*. Therefore, in the architecture design of DMFF, we decouple the relevant codes of the typification process as much as possible, so that the core computing code based on atomic parameters has its own independent API and can be called separately.

The design of the typing module of DMFF is basically based on the existing framework of OpenMM. DMFF needs to keep the derivation chain uninterrupted when unfolding force field params into atomic params. Therefore, under the condition of maintaining the same call logic and process of OpenMM, it rewrites the typing code of OpenMM with Jax. Generally speaking, OpenMM/DMFF requires users to clearly define the category of each atom in each residue and the connection mode between atoms to form a residue definition template. Then the residue template is used to match the PDB file to typify the whole system. See the following [documents](../dev_guide/arch.MD) for details.

## References：

1. [Anthony's book](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199672394.001.0001/acprof-9780199672394)
2. [The Multipolar Ewald paper in JCTC:  J. Chem. Theory Comput. 2015, 11, 2, 436–450](https://pubs.acs.org/doi/abs/10.1021/ct5007983)
3. [The dispersion Ewald/PME](https://aip.scitation.org/doi/pdf/10.1063/1.470117)
4. [Frenkel & Smit book](https://www.elsevier.com/books/understanding-molecular-simulation/frenkel/978-0-12-267351-1)
5. Note: multipole ewald.pdf
6. [MPID Reference](https://doi.org/10.1063/1.4984113)