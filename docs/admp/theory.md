# theory background

This project aims to implement an organic force field with a differentiable framework to automatically derivate atomic position, box shape, force field parameters, and other inputs. 

The ADMP force field module has following interactions：

* Non-bond interaction
* Polarization term
* Dispersion term
* Long-range interaction with PME
* Short-range pair interaction
* Topological scaling
* Neighborlist

## Non-bonded Interaction

### electrostatic term

For electrostatic interaction between electron could of atoms, we can use multipole expansion with cutoff. If only the charge (zero-moment) part is retained, it reduces to point charge model in classical force field:

$$V=\sum_{ij} \frac{q_i q_j}{r_{ij}}$$

where $$q_i$$ is charge number of atom.

More complex force field forms can be obtained by increasing truncation of the moment order. Some force fields, such as AMOEBA and MPID, use higher-order truncation. In DMFF, we have up to four moments:

$$V=\sum_{tu} \hat{Q}_t^A T^{AB}_{tu} \hat{Q}_u^B$$

where $$Q_t^A$$ represents t-component of multipole moment of atom A. There are two definations: cartesian coordinates and spherical harmonics. In DMFF, we use spherical harmonics. the sequence is:

$$0, 10, 1c, 1s, 20, 21c, 21s, 22c, 22s, ...$$

The $$T_{tu}^{AB}$$ represents interaction tensor among multipoles, which mathmatical expression can refer to Ref 1 appendix F. The convertion of between different multipole moment definitions and rotation rule can refer to Ref 1 & 5.

1. multipole moment coordinate system

Different from charge, multipole moment definition depends on coordinate system. What we use are mainly three:

  - global frame: coordinate system binds to the simulation box. It same for all atoms. We use this system to calculate charge density structure factor $$S(\vec{k})$$ in reciprocal space.
  - local frame: this system define coordinate of each atom by positions of its peripheral atoms, and then multipole moment is given under this system. Gerenally, atomic multipole moments have considerable stability in local frame, so it is more suitable as force field parameter input. In DMFF, the definition of local frame is exactly the same as AMOEBA plugin in OpenMM. The detail can refer to following literatures:
      * OpenMM source code, version 7.4, wrappers/python/simtk/openmm/app/forcefield.py, line 4568~4578
      * J. Chem. Theory Comput. 2013, 9, 9, 4046–4063 (https://pubs.acs.org/doi/abs/10.1021/ct4003702)
  - quasi internal frame, aka. QI frame: a special coordinate system to calculate interaction between two site in real space. Taking the connecting line of two sites as the Z-axis, the interaction tensor can be greatly simplified by using symmetry under this coordinate system to $$T_{tu}^{AB}$$.

2. Polarization term

DMFF supports polarizable force fields, that is the dipole moment of the atom can respond to the change of the external electric field. We support each atom not only has permanent multipoles $$\hat{Q}_t$$, but induced dipole $$U_{ind}$$. The interaction between induced-induced dipole and between induced-permanent dipole needs damping, which mathmatical experssion same as MPID(Ref 6). Specifically, each atom needs a thole parameter ($$a_i$$). When calculate interaction of polarizable sites, damping function will be introduced. Take $$U_{ind}$$-permanent charge interaction as an example, the definition of damping function is:

$$1-\left(1+a u+\frac{1}{2} a^{2} u^{2}\right) e^{-a u} \\ a=a_i + a_j \\ u=r_{ij}/\left(\alpha_i \alpha_j\right)^{1/6} $$
Other damping form between multipole moment can refer to Ref 6, table I.

We solve $$U_{ind}$$ by minimizing electrostatic energy. The energy can be written as follows:

$$V=V_{perm-perm}+V_{perm-ind}+V_{ind-ind}$$

the last two term relate to $$U_{ind}$$. Without introducing the non-linear polarization term(e.g. some force fields introduces $$U^4$$ to avoid polarization catastrophe), the last two terms actually is quadratic function of $$U_{ind}$$: 

$$V_{perm-ind}+V_{ind-ind}=U^TKU-FU$$

where off-diagonal term of $K$ matrix is induced-induced dipole interaction, the diagonal term is form energy of induced dipole($$\sum_i \frac{U_i^2}{2\alpha_i}$$); the $F$matrix represents permanent multipole - induced dipole interaction. We use gradient descent method to optimize energy to get $$U_{ind}$$.

In the current version, we temporarily assume that the polarizability is spherically symmetric, that is, the polarizability $$a_i$$ is a scalar, not a tensor. It is relatively simple to relax this restriction, just change the reciprocal of polarizability to the inverse of matrix when calculating the diagonal term of $K$ matrix.

3. Dispersion term

We assume that the dispersion between atoms can be described by the following expansion

$$V_{disp}=\sum_{ij}-\frac{C_{ij}^6}{r_{ij}^6}-\frac{C_{ij}^8}{r_{ij}^8}-\frac{C_{ij}^{10}}{r_{ij}^{10}}-...$$

where dispersion factor satisfy following combination rule:

$$C^n_{ij}=\sqrt{C_i^n C_j^n}$$

note that according to the perturbation theory, the dispersion terms should be continuous even powers, so the odd dispersion terms are not supported in DMFF.

4. long-range interaction with PME

The long-range interaction will be involved in the treatment of electrostatic, polarization and dispersion interactions. Take charge-charge interaction as an example. The interaction decays in the form of $$O(\frac{1}{r})$$, and its energy does not converge with the increase of cutoff-distance. In the calculation of multipole and dispersion forces, the convergence speed of cutoff distance is also slow. We introduce Paricle Meshed Ewald(PME) method to calculate those interaction.

An example as charge-charge interaction, we split interaction tensor by long and short range in PME:

$$\frac{1}{r}=\frac{erfc(\kappa r)}{r}+\frac{erf(\kappa r)}{r}$$

The first term is a short-range term, which can be calculated directly by using a simple distance cutoff in real space. The second term is a long-range term, which needs to be calculated in reciprocal space by fast Fourier transform(FFT). The total energy of charge-charge interaction is:

$$E_{real}=\sum_{ij}\frac{erfc(\kappa r_{ij})}{r_{ij}}\\E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}\exp\left[\frac{k^2}{4\kappa^2}\right]\left|S(\vec{k})\right|^2}\frac{1}{\left|\theta(\vec{k})\right|^2}\\E_{self}=-\frac{\kappa}{\sqrt{\pi}}\sum_i {q_i^2} \\E=E_{real}+E_{recip}+E_{self}$$

As for multipole PME and dispersion PME, the expression refer to Ref 2, Ref 3 and Ref 5.

The key parameters in PME include:

  - $$\kappa$$: controls the separation degree of long-range and short-range. The greater $$\kappa$$, the faster decay of the real space, the smaller the cutoff distance that can be used in the real space, and the more difficult the convergence of the reciprocal space and the more k-points are required;

  - $$ r_{c}$$: cutoff distance in real space;

  - $$ K_{max}$$: controls the number of maximum k-points


In DMFF, we determine these parameters in the same way as the PME in [OpenMM](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald):

$$\kappa=\sqrt{-\log (2 \delta)} / r_{c} \\ K_{max}=\frac{2 \kappa d}{3 d^{1 / 5}}$$

Where, the user needs to specify the cutoff distance $$r_c$$ when building the neighbor list, the width of the box in each dimension $$d$$, and the energy calculation accuracy $$\delta$$.

In the current version, the parameter determination mechanism of multipole PME and dispersion PME is exactly the same as that of charge PME.

5. short-range interaction

Short distance pair interaction refers to all interactions with the following forms:

$$V=\sum_{ij}v(r_{ij})$$

Some common short-range pair interactions include:

  -  Buckingham or excluded part of LJ potential:

$$v(r)=A\exp(-\beta r)\\v(r)=\frac{C^{12}}{r^{12}}$$

  - Tang-Tonnies Damping：Damping function for short-range electrostatic and dispersive effects
    1. Combination Rule:

For most traditional force fields, pairwise parameters in each pair of interactions are determined by atomic parameters. This mathematical relationship is called combination rule. For example, in the calculation of LJ potential, the following combination rule may be used:    

$$v(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]\\ \sigma=\frac{\sigma_i + \sigma_j}{2} \\ \varepsilon=\sqrt{\varepsilon_i \varepsilon_j}$$

DMFF does not make any assumptions about the specific mathematical forms of combination rule and $v(r)$. Users need to write them in the definition of pairwise kernel function.

6. neighbor list

The calculation of all real spaces depends on the construction of neighbor list. Its purpose is to find the "nearest neighbor" within a certain distance of the central atom by using an efficient method and calculate its interaction.

In DMFF, we use external code([jax-md](https://github.com/google/jax-md)) to build the nearest neighborlist. In all real space code, an external input argument named pairs is required, which contains the sequence numbers of all atomic pairs within a certain distance $$r_c$$. We assume that the pairs variable is in the form of `sparse` or `ordered sparse` in jax-md.

Because the nearest neighborlist only provides atom **id** information, it has nothing to do with the derivation chain.

7. Topological scaling

In the organic molecular force field, in order to avoid double counting with the bonding interaction term, we generally need to scale the non-bonding interaction between two atoms link to each other in topology. The specific scaling degree depends on the topological distance between the two atoms. We generally define two atoms with one bond as "1-2" interaction, and those separated by two bonds as "1-3" interaction, and so on. For example, in the OPLS-AA force field, all "1-2" non-bonding actions are turned off, while all "1-3" non-bonding actions are scaled to 50% of the normal non-bond actions.

Important variables related to topological scaling in DMFF include:

  - `covalent_Map`: the format is N*N matrix, which defines the topological spacing between any two atoms. If the matrix element is 0, it indicates that the **topological** distance between the two atoms is too far, so there is a complete non-bonding interaction between them.

  - `mScales`: The sequence of scaling factors is stored. The first element is 1-2 non-bond scaling, the second element is 1-3 non-bond scaling, and so on. The last number of the sequence **must be 1**, which represents the complete, unscaled non-bond interaction.

  - `pScales`/`dScales`: It is only related to the calculation of polarization energy, representing induced-perm and induced-induced respectively. The format is the same as that of `mScales`.


8. General multibody terms (such as ML force field)

  TODO:



2. 成键相互作用
分子内成键相互作用指所有依赖于分子内坐标（internal coordinates，IC）的相互作用，内坐标主要包括：键长、键角、二面角等。
  1. Harmonic Bonding Terms
DMFF中bonding term的定义与OpenMM中相同，对于每一根键，我们有：
$$E=\frac{1}{2}k(x-x_0)^2$$
注意力常数之前的1/2系数
  2. Harmonic Angle Terms
对于键角，我们同样有：
$$E=\frac{1}{2} k\left(\theta-\theta_{0}\right)^{2}$$
  3. Dihedral Terms
    1. Proper Dihedral
    2. Improper Dihedral
  4. 多IC耦合项

3. 原子分类（typification）
在能量计算开始前，原子和IC参数（如电荷、多极矩、色散系数、极化率、每根键、键角的力常数等）均需要赋值。一般而言，这些参数都应和每个原子或IC的具体环境和构象有关。但在传统力场中，为减少参数个数，一般将原子和IC按照其拓扑环境进行归类，同类原子或IC共享参数。对每个原子和IC进行归类，并对其参数进行赋值的过程被称为typification。

在DMFF中，我们将需要优化的输入参数称为力场参数（force field parameters），并将完成typification之后每个原子和IC的参数称为原子参数（atomic parameters）。注意在理想的新型力场中，如果我们可以采用机器学习模型直接预测atomic parameters，则typification的过程不是必须的。因此在DMFF的架构设计中，我们尽可能解耦了typification过程的相关代码，使得基于atomic parameters的核心计算代码有自己独立的API，可以单独被调用。

DMFF的typification模块的设计基本基于OpenMM的现有框架。DMFF在将force field params展开为atomic params时，需要维持求导链不中断，因此在维持OpenMM调用逻辑和流程不变的条件下，使用JAX改写了OpenMM的typification代码。大体而言，OpenMM/DMFF需要用户对每个残基中的每个原子的类别，以及原子间的连接方式进行清晰定义，形成残基定义模版。然后利用残基模版匹配pdb文件，从而对整个体系进行typification。具体参见下列文档：
OpenMM XML力场/拓扑文件结构整理 

10. Atomic classification

Before energy calculation, atomic and IC parameters (such as charge, multipole moment, dispersion coefficient, polarizability, force constant of each bond and angle, etc.) need to be assigned. Generally speaking, these parameters should be related to the specific environment and conformation of each atom or IC. However, in the traditional force field, in order to reduce the number of parameters, atoms and ICs are generally classified according to their topological environment, and atoms in same class or ICs share parameters. The process of classifying each atom and IC and assigning its parameters is called typification.

In DMFF, the input parameters that need to be optimized are called **force field parameters**, and the parameters of each atom and IC after typing are called **atomic parameters**. Note that in the new ideal force field, if we can directly predict atomic parameters using machine learning model, the process of typification is *not necessary*. Therefore, in the architecture design of DMFF, we decouple the relevant codes of the typification process as much as possible, so that the core computing code based on atomic parameters has its own independent API and can be called separately.

The design of the typing module of DMFF is basically based on the existing framework of OpenMM. DMFF needs to keep the derivation chain uninterrupted when unfolding force field params into atomic params. Therefore, under the condition of maintaining the same call logic and process of OpenMM, it rewrites the typing code of openmm with Jax. Generally speaking, openmm/DMFF requires users to clearly define the category of each atom in each residue and the connection mode between atoms to form a residue definition template. Then the residue template is used to match the PDB file, so as to typification the whole system. See the following [documents](../dev_guide/arch.md) for details.



References：

1. [Anthony's book](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199672394.001.0001/acprof-9780199672394)
2. [The Multipolar Ewald paper in JCTC:  J. Chem. Theory Comput. 2015, 11, 2, 436–450](https://pubs.acs.org/doi/abs/10.1021/ct5007983)
3. [The dispersion Ewald/PME](https://aip.scitation.org/doi/pdf/10.1063/1.470117)
4. [Frenkel & Smit book](https://www.elsevier.com/books/understanding-molecular-simulation/frenkel/978-0-12-267351-1)
5. Note：multipole ewald.pdf
6. [MPID Reference](https://doi.org/10.1063/1.4984113)