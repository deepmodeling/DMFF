# 理论背景
参考文献：
1. Anthony's book: https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199672394.001.0001/acprof-9780199672394
2. The Multipolar Ewald paper in JCTC:  J. Chem. Theory Comput. 2015, 11, 2, 436–450 (https://pubs.acs.org/doi/abs/10.1021/ct5007983)
3. The dispersion Ewald/PME: https://aip.scitation.org/doi/pdf/10.1063/1.470117
4. Frenkel & Smit book: https://www.elsevier.com/books/understanding-molecular-simulation/frenkel/978-0-12-267351-1
5. 开发笔记：暂时无法在文档外展示此内容
6. MPID Reference:  https://doi.org/10.1063/1.4984113

本项目旨在采用可微分编程框架实现有机分子力场的计算，并实现对原子位置、盒子形状、力场参数等输入的自动求导。
MDFF可以计算的有机分子力场包括以下部分：
1. 非键相互作用
  1. 静电项
原子电子云间的静电相互作用，我们可以对每个原子的电子云进行多极展开，并进行截断，如仅保留电荷（0极矩）部分，可以得到传统力场中常用的点电荷模型：
$$V=\sum_{ij} \frac{q_i q_j}{r_{ij}}$$
其中$$q_i$$表示原子电荷数。

截断阶数提高，可以得到更加复杂的力场形式，AMOEBA、MPID等力场均采用了更高阶的截断。在DMFF中，我们最高保留至四极矩：
$$V=\sum_{tu} \hat{Q}_t^A T^{AB}_{tu} \hat{Q}_u^B$$
其中，$$Q_t^A$$代表A原子的多极矩的t-分量，其定义方式有两种：笛卡尔坐标和球谐函数，在DMFF代码内部我们采用球谐函数方式定义，其顺序默认如下：
$$0, 10, 1c, 1s, 20, 21c, 21s, 22c, 22s, ...$$
而$$T_{tu}^{AB}$$代表多极矩间的相互作用张量。其具体数学形式可参照Ref 1. 附表F。不同多极矩定义方式间的转化和旋转规则可以参考Ref 1 & 5。
    1. 多极矩坐标系
与电荷不同，多极矩的定义依赖于坐标系，我们经常采用的坐标系有三种：
    - 全局坐标（global frame）：即盒子本身的坐标系，对于所有原子而言均相同。我们一般在倒易空间计算电荷密度结构因子（$$S(\vec{k})$$）时采用该坐标。
    - 局域坐标（local frame）：采用周边原子的位置来定义每个原子的局域坐标，进而在该坐标下给出原子的多极矩。一般而言，原子多极矩在局域坐标下具有相当的稳定性，因此局域坐标比较适用于力场参数输入。在MDFF项目中，对于局域坐标系的定义，我们采用了和OpenMM中AMOEBA plugin完全相同的方式，具体细节可以参考如下文献：
      - OpenMM source code, version 7.4, wrappers/python/simtk/openmm/app/forcefield.py, line 4568~4578
      - J. Chem. Theory Comput. 2013, 9, 9, 4046–4063 (https://pubs.acs.org/doi/abs/10.1021/ct4003702)
    - 准内部坐标（Quasi Internal Frame，QI frame）：在实空间中计算两个site间的相互作用时采用的特殊坐标系，以两个site的连线为z轴，在该坐标系下可以利用对称性极大简化相互作用张量$$T_{tu}^{AB}$$的数学形式。

  2. 极化项
MDFF支持可极化力场，也即原子的偶极矩可以响应外电场的变化。具体而言，我们假定每个原子除固定多极矩$$\hat{Q}_t$$（permanent multipoles）之外，还具有诱导偶极$$U_{ind}$$，其大小由外场与原子极化率$$\alpha_i$$共同决定。为避免极化灾难（polarization catastrophe，即诱导偶极相互增强导致不收敛），诱导偶极间以及诱导偶极和固定多极间的相互作用需要damping，MDFF中damping的具体数学形式与MPID相同（Ref 6）。具体而言，对于每一个原子需要指定一个thole parameter ($$a_i$$）。当计算polarizable sites间的相互作用时，我们需要引入damping function，以$$U_{ind}$$- permanent charge相互作用为例，其damping function定义如下：
$$1-\left(1+a u+\frac{1}{2} a^{2} u^{2}\right) e^{-a u} \\ a=a_i + a_j \\ u=r_{ij}/\left(\alpha_i \alpha_j\right)^{1/6} $$
其他各极矩间的damping形式可以参考Ref. 6 Table I。

在MDFF中，我们采用最小化静电能的方式求解$$U_{ind}$$，静电能可以写成如下形式：
$$V=V_{perm-perm}+V_{perm-ind}+V_{ind-ind}$$
其中后两项与$$U_{ind}$$有关。在不引入非线性极化项（例如，某些力场为避免极化灾难会引入$$U^4$$项）的情况下，后两项实际上是$$U_{ind}$$的二次函数：
$$V_{perm-ind}+V_{ind-ind}=U^TKU-FU$$
其中K矩阵的非对角项是induced dipole - induced dipole相互作用，其对角项为induced dipole形成能$$\sum_i \frac{U_i^2}{2\alpha_i}$$，而F矩阵代表permanent multipole - induced dipole相互作用。在DMFF中，我们采用梯度下降法优化能量并得到$$U_{ind}$$。

在当前版本中，我们暂时假定极化率是球对称的，即极化率$$a_i$$为标量而非张量。放松这一限制较为简单，只需在计算K矩阵的对角项时将极化率倒数改为矩阵逆即可。

  3. 色散项
在MDFF中，我们假定原子间的色散作用可以采用如下展开描述：
$$V_{disp}=\sum_{ij}-\frac{C_{ij}^6}{r_{ij}^6}-\frac{C_{ij}^8}{r_{ij}^8}-\frac{C_{ij}^{10}}{r_{ij}^{10}}-...$$
其中色散系数满足如下combination rule：
$$C^n_{ij}=\sqrt{C_i^n C_j^n}$$
注意按照微扰理论，色散项应均为连续的偶数次幂，因此在MDFF中，暂不支持奇数次的色散项。


  4. 长程作用与PME
在静电、极化、色散三种相互作用的处理中会涉及长程相互作用。以charge-charge interaction为例，该相互作用以$$O(\frac{1}{r})$$的形式衰减，其能量随cutoff-distance的增长并不收敛。在多极、色散力的计算时，cutoff-distance的收敛速度也较慢。因此，在DMFF中我们可以采用Particle Meshed Ewald (PME)技术计算这些相互作用。
以charge-charge interaction为例，在PME中我们对其相互作用张量进行长短程拆分：
$$\frac{1}{r}=\frac{erfc(\kappa r)}{r}+\frac{erf(\kappa r)}{r}$$
其中第一项为短程项，可以在实空间中采用简单的distance cutoff直接计算。第二项为长程项，需要通过快速傅里叶变换在倒易空间中计算。charge-charge interaction PME能量的最终形式为：
$$E_{real}=\sum_{ij}\frac{erfc(\kappa r_{ij})}{r_{ij}}\\E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}\exp\left[\frac{k^2}{4\kappa^2}\right]\left|S(\vec{k})\right|^2}\frac{1}{\left|\theta(\vec{k})\right|^2}\\E_{self}=-\frac{\kappa}{\sqrt{\pi}}\sum_i {q_i^2} \\E=E_{real}+E_{recip}+E_{self}$$
对于multipole PME和dispersion PME，具体数学细节请分别参见Ref 2和Ref 3，以及Ref 5。

PME中的关键参量包括：
  - $$\kappa$$：该变量控制着长短程的分离程度，$$\kappa$$越大，则实空间部分衰减越快，实空间部分可以采用的cutoff distance越小，同时倒易空间部分收敛越难，需要的k点越多。
  - $$ r_{c}$$：实空间的cutoff distance
  - $$ K_{max}$$：该变量控制着最大K点的数量
在MDFF中，我们采用与OpenMM中静电PME相同的方式决定这些参量：
$$\kappa=\sqrt{-\log (2 \delta)} / r_{c} \\ K_{max}=\frac{2 \kappa d}{3 d^{1 / 5}}$$
其中，用户需要指定构建neighbor list时所使用的cutoff distance $$ r_{c}$$，盒子在每个dimension的宽度$$d$$，以及能量计算精度$$\delta$$。

在当前版本中，multipole PME和dispersion PME的参数决定机制与charge PME完全相同。

  5. 短距对相互作用
短距对相互作用是指所有具有如下形式的相互作用：
$$V=\sum_{ij}v(r_{ij})$$
一些常见的短距对相互作用包括：
  - Buckingham或LJ potential的排斥部分：
$$v(r)=A\exp(-\beta r)\\v(r)=\frac{C^{12}}{r^{12}}$$
  - Tang-Tonnies Damping：对短距静电和色散作用的damping function
    1. Combination Rule
对于绝大多数传统力场而言，每一对相互作用中的pairwise参数由原子参数决定，这一数学关系称之为combination rule，例如：在LJ potential的计算中，可能采用如下combination rule:
$$v(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]\\ \sigma=\frac{\sigma_i + \sigma_j}{2} \\ \varepsilon=\sqrt{\varepsilon_i \varepsilon_j}$$
DMFF不对Combination Rule和v(r)的具体数学形式做任何假定，用户需要在pairwise kernel函数的定义中自行写出。

  6. 近邻表
所有实空间的计算都依赖近邻表（neighbor list）的构建，其目的是采用较高效的方法寻找中心原子一定距离内的“近邻”，并计算其相互作用。
在DMFF中，我们采用外部代码（jax-md）来构建近邻表。在所有实空间的代码中，都需要名为pairs的外部输入变量，其中包含了所有一定距离范围（$$r_c$$）内的原子对序号。我们假定pairs变量的格式为jax-md中的Sparse或Ordered Sparse格式。
因为近邻表只提供原子序号信息，因此与求导链无关。

  7. Topological scaling
在有机分子力场中，为避免与成键相互作用项间的double counting，我们一般需要对拓扑上距离较近的两个原子间的非键作用进行scaling。具体的scaling程度取决于两个原子间的拓扑间距。我们一般将相互间距为一根键的两个原子定义为1-2作用，间距为两根键的定义为1-3作用，等等。比如，在OPLS-AA力场中，所有1-2非键作用均被关闭，而所有1-3非键作用均被scale为正常非键作用的1/2。
DMFF中与Topological scaling相关的重要变量包括：
  - Covalent_map: 格式为N*N矩阵，定义了任意两个原子间的拓扑间距。如矩阵元为0，则表明两个原子间在拓扑上距离过远，因此两者间具有完整的非键作用。
  - mScales：存储了scaling factor的数列，第一个元素是1-2非键作用的scaling，第二个元素是1-3非键作用的scaling，等等。该数列最后一个数必须为1，代表完整的，unscaled的非键作用。
例如，对于OPLS-AA力场，mScales的值应为：[0.0, 0.5, 1.0]
  - pScales/dScales：仅与极化能量的计算相关，分别代表induced-perm，以及induced-induced间的scaling，格式与mScales相同。

  8. 一般多体项（如ML力场）
暂略

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
