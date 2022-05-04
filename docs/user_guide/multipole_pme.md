# Notes on Multipolar Ewald



## References:

* [Anthony's book](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199672394.001.0001/acprof-9780199672394)
* The Multipolar Ewald paper in JCTC:  ***J. Chem. Theory Comput.*** 2015, **11**, 2, 436–450 [Link](https://pubs.acs.org/doi/abs/10.1021/ct5007983)
* The `convert_mom_to_xml.py` code, which converts the CAMCASP moments (global harmonics) to OpenMM moments (local cartesian).
* The dispersion Ewald/PME: [Link](https://aip.scitation.org/doi/pdf/10.1063/1.470117)



## Relevant mathematics

### Cartesian Tensors

Multipole operators in Cartesian Tensors:

$$
\displaylines{
\begin{align}
&\hat{q} = 1 \\
&\hat{\mu}_{\alpha} = r_{\alpha} \\
&\hat{\Theta}_{\alpha\beta}=\frac{3}{2}\left(r_{\alpha}r_{\beta}-\frac{1}{3}r^2\delta_{\alpha\beta}\right)
\end{align}
}
$$

Interaction Hamiltonian (up to quadrupole):

$$
\displaylines{
\hat{H}' = T q^{A} q^{B}+T_{\alpha}\left(q^{A} \hat{\mu}_{\alpha}^{B}-\hat{\mu}_{\alpha}^{A} q^{B}\right)+T_{\alpha \beta}\left(\frac{1}{3} q^{A} \hat{\Theta}_{\alpha \beta}^{B}-\hat{\mu}_{\alpha}^{A} \hat{\mu}_{\beta}^{B}+\frac{1}{3} \hat{\Theta}_{\alpha \beta}^{A} q^{B}\right) \\
-\frac{1}{3} T_{\alpha \beta \gamma}\left(\hat{\mu}_{\alpha}^{A} \hat{\Theta}_{\beta \gamma}^{B}-\hat{\Theta}_{\alpha \beta}^{A} \hat{\mu}_{\gamma}^{B}\right) +\frac{1}{9}T_{\alpha \beta \gamma \delta}\hat{\Theta}_{\alpha \beta}^{A} \hat{\Theta}_{\gamma \delta}^{B} + \cdots
}
$$

And the corresponding interaction tensors are:

$$
T_{\alpha \beta \ldots v}^{(n)}=\frac{1}{4 \pi \epsilon_{0}} \nabla_{\alpha} \nabla_{\beta} \ldots \nabla_{v} \frac{1}{R}
$$

The first few terms are listed in the page 4 of Anthony's book (up to quad-quad interactions).

### Spherical Tensors

The often seen spherical harmonics include:

* The original spherical harmonics (**adopting the convention without $(-1)^m$**):

$$
Y_{lm}(\theta, \phi) = \sqrt{\frac{2 l+1}{4 \pi} \frac{(l-m) !}{(l+m) !}} P_{l}^{m}(\cos \theta) e^{i m \phi}
$$

  with $P_l^m$ being the associated Legendre Polynomials:
$$
P_{\ell}^{m}(x)=\frac{(-1)^{m}}{2^{\ell} \ell !}\left(1-x^{2}\right)^{m / 2} \frac{d^{\ell+m}}{d x^{\ell+m}}\left(x^{2}-1\right)^{\ell}
$$
  Note we have:
$$
P_{\ell}^{-m}(x)=(-1)^{m} \frac{(\ell-m) !}{(\ell+m) !} P_{\ell}^{m}(x)
$$
  The exact form of these polynomials can be found in: 

  [Associated_Legendre_polynomials](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials)

  The lowest few orders of them are:

$$
\displaylines{
\begin{aligned}
&P_{0}^{0}(x)=1 \\
&P_{1}^{0}(x)=x \\
&P_{1}^{1}(x)=-\left(1-x^{2}\right)^{1 / 2} \\
&P_{2}^{0}(x)=\frac{1}{2}\left(3 x^{2}-1\right) \\
&P_{2}^{1}(x)=-3 x\left(1-x^{2}\right)^{1 / 2} \\
&P_{2}^{2}(x)=3\left(1-x^{2}\right)
\end{aligned}
}
$$

* We also use the renormalized spherical harmonics:

$$
\displaylines{
C_{l m}(\theta, \varphi)=\sqrt{\frac{4 \pi}{2 l+1}} Y_{l m}(\theta, \varphi) \\
 = \sqrt{\frac{(l-m) !}{(l+m) !}} P_{l}^{m}(\cos \theta) e^{i m \phi} \\
 =\epsilon_m\sqrt{\frac{(l-|m|)!}{(l+|m|)!}} P_l^{|m|}(\cos\theta)e^{im\phi}
}
$$

**Note that I believe the equation B.1.3 in Pg. 272 of Anthony's book is not quite right** (missing an absolute value in $P_l^m$). Definition of $\epsilon_m$ can be found in Pg. 272.

* Now we can define the regular and irregular spherical harmonics:

$$
\displaylines{
\begin{aligned}
R_{l m}(\vec{r}) &=r^{l} C_{l m}(\theta, \varphi) \\
I_{l m}(\vec{r}) &=r^{-l-1} C_{l m}(\theta, \varphi)
\end{aligned}
}
$$

In particular:

$$
R_{lm}(\vec{r}) = r^l \cdot \sqrt{\frac{(l-m) !}{(l+m) !}} P_{l}^{m}(\cos \theta) e^{i m \phi}
$$

From this we can define the real-valued regular spherical harmonics (for $m>0$):

$$
\displaylines{
\begin{aligned}
R_{l m c} &=\sqrt{\frac{1}{2}}\left[(-1)^{m} R_{l m}+R_{l,-m}\right] \\
\mathrm{i} R_{l m s} &=\sqrt{\frac{1}{2}}\left[(-1)^{m} R_{l m}-R_{l,-m}\right]
\end{aligned}
}
$$

According to my derivation, that is:

$$
\displaylines{
\begin{cases}
R_{l0} = r^l\cdot C_{l0} = r^l\cdot P_l^{0}(\cos\theta)\\
R_{lmc} = (-1)^m \cdot \sqrt{2} \cdot r^l \sqrt{\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)\cos(m\phi) \\
R_{lms} = (-1)^m \cdot \sqrt{2} \cdot r^l \sqrt{\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)\sin(m\phi)
\end{cases}
}
$$

In Anthony's book, they usually use Greek letters ($\mu,\nu,\kappa$ etc.​) or letter $t,u$ to represent $1c, 1s, 2c, 2s$​ etc. These are the projector functions to compute all the moments.

In the real-valued regular spherical harmonics representation, the interaction energy is:

$$
\hat{H}' = \sum_{at,bu} {\hat{Q}_t^a T_{tu}^{ab}\hat{Q}_u^b}
$$

The interaction tensor $T_{tu}^{AB}$ can be found in Appendix F of Anthony's book (pg. 291).

* A couple of things regarding paper: ***J. Chem. Theory Comput.*** 2015, **11**, 2, 436–450 (https://pubs.acs.org/doi/abs/10.1021/ct5007983)

In this paper, the definition (***in their notation***) of:

$$
\displaylines{
C_{l\mu} = (-1)^{\mu}\sqrt{\left(2-\delta_{\mu, 0}\right)(l+\mu) !(l-\mu) !}\cdot R_{l\mu} \\
=
\begin{cases}
(-1)^{m}\sqrt{\left(2-\delta_{m, 0}\right)(l+m) !(l-m) !} \cdot R_{l mc}(\vec{r}) & \mu \geq 0 \\ 
(-1)^{-m}\sqrt{2(l-m) !(l+m) !} \cdot R_{lms}(\vec{r}) & \mu<0
\end{cases} \\
(\text{Here, we assumes $m=|\mu|$}) \\
=
\begin{cases}
r^l\cdot P_l^0(\cos\theta) & \mu=0 \\
(-1)^m\cdot\sqrt{2}\cdot r^l\sqrt{\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)\cos(m\phi) & \mu\gt 0 \\
(-1)^m\cdot\sqrt{2}\cdot r^l\sqrt{\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)\sin(m\phi) & \mu\lt 0
\end{cases}
}
$$

This is identical to Eqn. 14. **Therefore, the $C_{l\mu}$ in the JCTC paper is identical to the $R_\mu$ in Anthony's book**.

And the interaction function also checks out! So we have a compact form for the interaction tensor in table F.1 of the book

$$
T_{tu}^{ab} = \frac{R_t(\nabla_a)}{(2l_t-1)!!}\frac{R_u(\nabla_b)}{(2l_u-1)!!}\frac{1}{R_{ab}}
$$

This is verified manually by computing the $T_{21c,11c}=T_{xz, x} = \sqrt{3}\frac{1}{R^4}\left(\hat{z}-5\hat{x}^2\hat{z}\right)$​ tensor.



## Rotations

Assume the local frame matrix for a particular atom is:

$$
\displaylines{
\mathbf{R} = 
\left[
\matrix{
\mathbf{r_1^T} \\
\mathbf{r_2^T} \\
\mathbf{r_3^T} \\
}\right] = \left[
\matrix{
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2 \\
x_3 & y_3 & z_3 
}\right]
}
$$

Then for any vector, the local ($\vec{r}'$)-to-global (\vec{r}) transform is:

$$
\vec{r}=\mathbf{R}^T\cdot\vec{r}'
$$

Following the convention in the `convert_mom_to_xml.py` code, assuming the order for spherical harmonics is: $Q_{00}, Q_{10}, Q_{11c}, Q_{11s}, Q_{20}, Q_{21c}, Q_{21s}, Q_{22c}, Q_{22s}$​​

* **Dipole rotation**

For dipole, the harmonics-to-cartesian conversion is:

$$
\mathbf{C}_1=\left[
\matrix{
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
}
\right]
$$

So the **local-harmonics-to-global-harmonics** conversion for dipole moments is:

$$
\mathbf{R}_{1}^{l-g}=\mathbf{C}_1^T\mathbf{R}^T\mathbf{C}_1
$$

* **Quadrupole rotation**

First of all, the cartesian moments are:

$$
\displaylines{
\mathbf{\Theta}=
\int {d\vec{r}\cdot\rho(\vec{r})\cdot\left(\frac{3}{2}\left[\matrix{
xx & xy & xz \\
yx & yy & yz \\
zx & zy & zz \\
}\right] - \frac{1}{2}r^2\mathbf{I} \right)} \\
= \int {d\vec{r}\cdot\rho(\vec{r})\cdot\left(\frac{3}{2}\vec{r}\vec{r}^T -\frac{1}{2}(\vec{r}^T\cdot\vec{r})\mathbf{I}\right)}
}
$$

Then obviously, the **local-to-global** conversion is:

$$
\mathbf{\Theta} = \mathbf{R}^T\mathbf{\Theta}'\mathbf{R}
$$

The conversion between $\vec{\Theta}=[xx,yy,zz,xy,xz,yz]$​​​​ and harmonic moments ($\vec{Q}=[Q_{20}, Q_{21c}, Q_{21s}, Q_{22c}, Q_{22s}]$​​​​​) is:

$$
\displaylines{
\mathbf{C}_2^{h-c} = \left[\matrix{
-\frac{1}{2} & 0 & 0 & \frac{\sqrt{3}}{2} & 0 \\
-\frac{1}{2} & 0 & 0 & -\frac{\sqrt{3}}{2} & 0 \\
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \frac{\sqrt{3}}{2} \\
0 & \frac{\sqrt{3}}{2} & 0 & 0 & 0 \\
0 & 0 & \frac{\sqrt{3}}{2} & 0 & 0
}\right]
}
$$

$$
\vec{\Theta} = \mathbf{C}_2^{h-c} \vec{Q}
$$

And the inversed conversion is:

$$
\displaylines{
\mathbf{C}_2^{c-h} = \left[\matrix{
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \frac{2}{\sqrt{3}} & 0 \\
0 & 0 & 0 & 0 & 0 & \frac{2}{\sqrt{3}} \\
\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{3}} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & \frac{2}{\sqrt{3}} & 0 & 0 \\
}\right]
}
$$

$$
\vec{Q} = \mathbf{C}_2^{c-h}\vec{\Theta}
$$

So the rotation procedure from **local-to-global** rotation for (spherical harmonics) quadrupole moments is:

$$
\displaylines{
\vec{\Theta}' = \mathbf{C}_2^{h-c} \vec{Q}' \\
\mathbf{\Theta} = \mathbf{R}^T\mathbf{\Theta}'\mathbf{R} \\
\vec{Q} = \mathbf{C}_2^{c-h}\vec{\Theta}
}
$$

And vice-versa for the invert process.

Using `sympy`, we can derive the actual linear transformation matrix, which is coded in `test_rot_quad.py`. The derivation is done using `derive_rot_mat.py`





## Multipolar Ewald

* **The reciprocal space part:**

  The structure factor is:

  $$
  S(\vec{k}) = \sum_{a,t} Q_t^a\frac{R_t(\vec{k})(-i)^{l_t}}{(2l-1)!!}e^{-i\vec{k}\cdot\vec{R}_a}
  $$

  The reciprocal space energy is:

  $$
  E_{recip} = \sum_{k\neq 0} {\frac{2\pi}{V k^2}\exp\left(-\frac{k^2}{4\kappa^2}\right)\left|S(\vec{k})\right|^2}
  $$
  
* The real space part:

$$
\displaylines{
E_{real} = \sum_{\substack{a<b \\ t\in a, u\in b}} {Q_t^a \tilde{T}_{tu}^{ab} Q_u^b}
}
$$

​    And the damped real space operator is (**to be derived!**):

$$
\displaylines{
\tilde{T}_{t, u}^{a, b}=\frac{C_{t}\left(\nabla_{a}\right)}{(2 l_t-1) ! !} \frac{C_u\left(\nabla_{b}\right)}{(2 l_u-1) ! !} \frac{\operatorname{erfc}\left(\kappa R_{a b}\right)}{R_{a b}}
}
$$

* The self energy is:

$$
E_{self} = - \sum_{a, l \mu \in a} {Q_{l \mu}^a}^2 \sqrt{\frac{\kappa^2}{\pi}} \frac{(2 \kappa^2)^{l}}{(2 l+1) ! !}
$$

## Multipolar PME

* The real space and the self parta are the same as ewald
* The reciprocal space part, the structural factor is:

$$
\displaylines{
\begin{align}
S(\vec{k}) & = \sum_{\vec{m}} {\exp\left[-2\pi i\left(\frac{m_1k_1}{K_1}+ \frac{m_2k_2}{K_2} + \frac{m_3k_3}{K_3}\right)\right]}
\sum_{a,l\mu} {Q_{l\mu}^a \frac{R_{l\mu}(\nabla_a)}{(2l-1)!!}\theta(\vec{R_m}-\vec{R}_a)} \\
& =FFT\left(\sum_{a,l\mu} {Q_{l\mu}^a \frac{R_{l\mu}(\nabla_a)}{(2l-1)!!}\theta(\vec{R}_m-\vec{R}_a)}\right) \\
& =FFT\left(Q(\vec{m})\right)
\end{align}
}
$$

Here, $Q(\vec{m})$ is the "spreading" of moments on real space grid.

And the energy is:

$$
\displaylines{
E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}\exp\left[\frac{k^2}{4\kappa^2}\right]\left|S(\vec{k})\right|^2}\frac{1}{\left|\theta(\vec{k})\right|^2} \\
\theta(\vec{k}) = \theta(k_1)\theta(k_2)\theta(k_3) \\
\theta(k) = \sum_{m=-n/2}^{n/2} {M_n\left(m+\frac{n}{2}\right)\exp\left(-2\pi i\frac{mk}{K}\right)}
}
$$

The expression for $M_6(u)$​​ and its derivatives (related to $R_{l\mu}(\nabla_a)\theta(\vec{R}_m-\vec{R}_a)$​​​) is (**dervied by `derive_theta.py` code**):

$$
\displaylines{
M_6(u)=\begin{cases}
\frac{u^{5}}{120}  & 0 \le u \le 1  \\
\frac{u^{5}}{120} - \frac{\left(u - 1\right)^{5}}{20}  & 1 \le u \le 2  \\
\frac{u^{5}}{120} + \frac{\left(u - 2\right)^{5}}{8} - \frac{\left(u - 1\right)^{5}}{20}  & 2 \le u \le 3  \\
\frac{u^{5}}{120} - \frac{\left(u - 3\right)^{5}}{6} + \frac{\left(u - 2\right)^{5}}{8} - \frac{\left(u - 1\right)^{5}}{20}  & 3 \le u \le 4  \\
\frac{u^{5}}{24} - u^{4} + \frac{19 u^{3}}{2} - \frac{89 u^{2}}{2} + \frac{409 u}{4} - \frac{1829}{20}  & 4 \le u \le 5  \\
- \frac{u^{5}}{120} + \frac{u^{4}}{4} - 3 u^{3} + 18 u^{2} - 54 u + \frac{324}{5}  & 5 \le u \le 6  \\
\end{cases}
}
$$

$$
\displaylines{
M'_6(u)=\begin{cases}
\frac{u^{4}}{24}  & 0 \le u \le 1  \\
\frac{u^{4}}{24} - \frac{\left(u - 1\right)^{4}}{4}  & 1 \le u \le 2  \\
\frac{u^{4}}{24} + \frac{5 \left(u - 2\right)^{4}}{8} - \frac{\left(u - 1\right)^{4}}{4}  & 2 \le u \le 3  \\
- \frac{5 u^{4}}{12} + 6 u^{3} - \frac{63 u^{2}}{2} + 71 u - \frac{231}{4}  & 3 \le u \le 4  \\
\frac{5 u^{4}}{24} - 4 u^{3} + \frac{57 u^{2}}{2} - 89 u + \frac{409}{4}  & 4 \le u \le 5  \\
- \frac{u^{4}}{24} + u^{3} - 9 u^{2} + 36 u - 54  & 5 \le u \le 6  \\
\end{cases}
}
$$

$$
\displaylines{
M''_6(u)=\begin{cases}
\frac{u^{3}}{6}  & 0 \le u \le 1  \\
\frac{u^{3}}{6} - \left(u - 1\right)^{3}  & 1 \le u \le 2  \\
\frac{5 u^{3}}{3} - 12 u^{2} + 27 u - 19  & 2 \le u \le 3  \\
- \frac{5 u^{3}}{3} + 18 u^{2} - 63 u + 71  & 3 \le u \le 4  \\
\frac{5 u^{3}}{6} - 12 u^{2} + 57 u - 89  & 4 \le u \le 5  \\
- \frac{u^{3}}{6} + 3 u^{2} - 18 u + 36  & 5 \le u \le 6  \\
\end{cases}
}
$$

And we have something like:

$$
\displaylines{
\partial_\alpha\partial_\beta \theta(\vec{R}_m-\vec{R}_a) =  (2\pi)^2 \frac{K_\alpha K_\beta}{L_\alpha L_\beta}
M'_\alpha M'_\beta M_\gamma \\
\partial_\alpha^2 \theta(\vec{R}_m-\vec{R}_a) =  (2\pi)^2 \left(\frac{K_\alpha}{L_\alpha}\right)^2
M''_\alpha M_\beta M_\gamma
}
$$

**Also, be aware of the position of $R_a$, taking derivative w.r.t $a$ (i.e., the $\nabla_a$) gives a $(-1)^l$ term!!!**



## FFP Method

* The real space part is the same as Ewald, but simply replacing $\kappa$ with $\kappa/\sqrt{2}$​.
* There is no self term.
* The reciprocal space part is computed as following:

  First define the gaussian multipole function

$$
\displaylines{
\chi_{l \mu}\left(\vec{r}-\vec{R}_{a}\right)=\frac{C_{l \mu}\left(\vec{r}-\vec{R}_{a}\right)}{(2 l-1) ! !}(2 \kappa^2)^{l}\left(\frac{\kappa^2}{\pi}\right)^{3 / 2} \mathrm{e}^{-\kappa^2\left|\vec{r}-\vec{R}_{a}\right|^{2}}
}
$$

​       Then, use this function to spread moments on real space grid:

$$
Q(\vec{m}) = \sum_{a,l\mu} {Q_{l\mu}^a} \chi_{l\mu}(\vec{R}_m - \vec{R}_a)
$$

​        Then do Fourier transform:

$$
\tilde{\rho}(\vec{k}) = \frac{V}{K}FFT(Q(\vec{m}))
$$

​        And the energy is:

$$
E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}}\left|\tilde{\rho}(\vec{k})\right|^2
$$

* This seems to be simpler to code, but maybe slower than PME.

## Notes on the MPID Code:

* NOTE! We may directly copy the real space code from MPID plugin (`reference/src/SimTKReference/MPIDReferenceForce.cpp`)! Just notice the key quantities in the `calculatePmeDirectElectrostaticPairIxn` are:
  * mScale: the scaling between permanent multipole interactions
  * pScale: the scaling between permanent and induced dipole
  * dScale: the scaling between induced-induced dipole (? not sure about this yet, seems to be equivalent to pScale)
  * `rInvVec[n]` is simply $\frac{1}{r^n}$​​
  * `alphaRVec[n]` is simply $\kappa^n r^n$
  * `X` is: $ 2\frac{\exp(-\kappa^2 r^2)}{\sqrt{\pi}}$​​

* Regarding why the final energy is computed as the **interactions between fixed multipoles and fixed multipoles + half induced dipoles**. It can be understood as this:

  The energy as a function of induced dipole moments ($\mathbf{\mu}$​) is effectively a quadratic function:

  $$
  E_{tot} = \frac{1}{2}\mathbf{\mu}^T\mathbf{K}\mathbf{\mu} - F^T\mathbf{\mu}
  $$

  The first term contains the energy penalty for being polarized (the diagonal $\frac{1}{2\alpha} \mu^2$​​​ terms), and also the off-diagonal induced dipole - induced dipole interactions. $\mathbf{K}$​ must be symmetric, and $F$​, as a vector, is effectively the permanent electric field on each site. The second term is the interactions between induced dipole and fixed multipoles

  Then we minimize $E_{tot}$​ with respect to $\mu$, we obtain:

  $$
  \mathbf{K\mu} = F
  $$

  Then we have the total energy:

  $$
  E_{tot} = \frac{1}{2}\mu^TF-F^T\mu=-\frac{1}{2} F^T\mu
  $$

  That is, exactly the interactions between fixed multipole with half induced dipole (**formally, without the induced dip - induced dip interaction!**).





## Notes on the Dispersion PME

### Ewald    

​    Another extension for PME is the PME for potential with form $C_iC_j/r^p$​​ (e.g., dispersion interactions). See the original derivation in the Appendix A of:

​    https://aip.scitation.org/doi/pdf/10.1063/1.470117

​    Assuming that $C_{ij}=\sqrt{C_iC_j}=c_ic_j$​​ (suppose we define $c_i=\sqrt{C_i}$​​)

​    Then the basic Ewald energy ($\sum_{i<j}c_ic_j/r^p$) equations are:

$$
\displaylines{
E_{real} = \sum_{i<j} \frac{c_ic_j}{r^p}g_p(\kappa r_{ij}) \\
E_{recip} = \frac{\pi^{3/2}\kappa^{p-3}}{2V} \sum_\vec{k} f_p\left(\frac{k}{2\kappa}\right)\left|S(\vec{k})\right|^2 \\
E_{self} =-\frac{\kappa^p}{p\Gamma(p/2)}\sum_i c_i^2
}
$$

In here, we have:

$$
\displaylines{
f_p(x) = \frac{2x^{p-3}}{\Gamma(p/2)}\int_x^{\infty}{s^{2-p}\exp(-s^2)ds} \\
g_p(x) = \frac{2}{\Gamma(p/2)}\int_x^{\infty}{s^{p-1}\exp(-s^2)ds} \\
S(\vec{k}) = \sum_i {c_i\exp(i\vec{k}\cdot\vec{r}_i)}
}
$$

* The real space part:

​    The real space part mainly depends on $g_p$​, for dispersion, we have the following recursive relationship for $g$:

$$
\displaylines{
\begin{align}
g_2(x) &= \exp(-x^2) \\
g_p(x) &= \frac{x^{p-2}}{\Gamma\left(\frac{p}2{}\right)}\exp(-x^2) + g_{p-2}(x)
\end{align}
}
$$

Therefore, we have:

$$
\displaylines{
\begin{align}
g_6 & = \left(1+x^2+\frac{1}{2}x^4\right)\exp(-x^2) \\
g_8 & = \left(1+x^2+\frac{1}{2}x^4 + \frac{1}{6}x^6\right)\exp(-x^2) \\
g_{10} & = \left(1+x^2+\frac{1}{2}x^4 + \frac{1}{6}x^6 + \frac{1}{24}x^8\right)\exp(-x^2)
\end{align}
}
$$

* The reciprocal space part:

  The recursive relation of $f_p$​ is:

$$
\displaylines{
\begin{align}
f_2(x) & = \frac{\sqrt{\pi}}{x} \text{erfc}(x) \\
f_p(x) & = \frac{2}{(p-3)\Gamma\left(\frac{p}{2}\right)}\exp(-x^2) - \frac{4 x^2}{(p-2)(p-3)}f_{p-2}(x)
\end{align}
}
$$

  Therefore, we have:

$$
\displaylines{
\begin{align}
f_6 & = \left(\frac{1}{3} - \frac{2}{3}x^2\right) \exp(-x^2) + \frac{2}{3}x^3 \sqrt{\pi}\cdot\text{erfc}(x) \\
f_8 & = \left(\frac{1}{15} - \frac{2}{45}x^2 +\frac{4}{45}x^4\right) \exp(-x^2) - \frac{4}{45}x^5 \sqrt{\pi}\cdot\text{erfc}(x) \\
f_{10} &= \left(\frac{1}{84} - \frac{1}{210}x^2 +\frac{1}{315}x^4 - \frac{2}{315}x^6\right) \exp(-x^2) + \frac{2}{315}x^7 \sqrt{\pi}\cdot\text{erfc}(x)
\end{align}
}
$$
  
  NOTE: compare to the $p=1$ (Coulombic interaction case), the $\vec{k} = 0$ term is well defined and should not be dropped!
  
* **So literally, you can simply take the monopole (charge) PME code (use $\sqrt{C_i}$ as charges), and do the following changes:**

  * In real space:

  $$
  \operatorname{erfc}(\kappa r) \rightarrow g_p(\kappa r)
  $$

  * In reciprocal space:

  $$
  \frac{2\pi}{Vk^2}\exp\left(-\frac{k^2}{4\kappa^2}\right) \rightarrow \frac{\pi^{3/2}\kappa^{p-3}}{2V}f_p\left(\frac{k}{2\kappa}\right)
  $$

  And include the $\Gamma$ point in the reciprocal space summation.

  * In self-energy:

  $$
  \frac{\kappa}{\sqrt{\pi}} \rightarrow \frac{\kappa^p}{p\Gamma(p/2)}
  $$

That is it!

