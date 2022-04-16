# Architecture of DMFF

![arch](../assets/arch.png)

The overall framework of DMFF can be divided into two parts: parser & typing and calculators. We usually refer to the former as the *frontend* and the latter as the *backend* for ease of description.

DMFF introduces different forms of force fields in a modular way. For any form of force field, it is divided into frontend modules and backend modules. The frontend module is responsible for input file parsing, molecular topology construction, atomic typification, and unfolding from the forcefield parameter layer to the atomic parameter layer; The backend module is the calculation core, which calculates the energy & force of the system at a time by particle positions and system properties.

In the frontend module design, DMFF reuses the frontend parser module from OpenMM and realizes the functions of topology analysis. All frontend modules are stored in `api.py`, called by the Hamiltonian class.

The backend module is usually an automatically differentiable computing module built using Jax. 

The structure of frontend and backend modules will be introduced in detail in the following documents.

## How frontend works

Frontend modules are stored in `api.py`. `Hamiltonian` class is the top-level class exposed to users by DMFF. `Hamiltonian` class reads the path of the XML file, parses the XML file, and calls different frontend modules according to the XML tags. The frontend module has the same form as OpenMM's generator [forcefield.py](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py). The `Generator` class takes the XML tag in and parse the parameters, initialize the backend calculator and provide the interface of energy calculation method.

When the user uses the DMFF, the only thing that needs to do is initialize the `Hamiltonian` class. In this process, `Hamiltonian` will automatically parse and initialize the corresponding potential function according to the tags in XML. The call logic is shown in the following chart. The box represents the command executed in Python script, and the rounded box represents the internal operation logic of OpenMM when executing the command.

![openmm_workflow](../assets/opemm_workflow.svg)

### Hamiltonian class

The hamiltonian class is the top-level frontend module, inheriting from the [forcefield class](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py) of OpenMM. It is responsible for parsing XML force field files and generating potential functions to calculate system energy for given topology information. First, the use method of a Hamiltonian class is given:

```python
H = Hamiltonian('forcefield.xml')
app.Topology.loadBondDefinitions("residues.xml")
pdb = app.PDBFile("waterbox_31ang.pdb")
rc = 4.0
# generator stores all force field parameters
generator = H.getGenerators()
disp_generator = generator[0]
pme_generator = generator[1]

pme_generator.lpol = True # debug
pme_generator.ref_dip = 'dipole_1024'
potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
# pot_fn is the actual energy calculator
pot_disp = potentials[0]
pot_pme = potentials[1]
```

`Hamiltonian` class performs the following operations during instantiation:

* read Residue tag in XML, generate Residue template;
* read AtomTypes tag, store AtomType of each atom;
* for each Force tag, call corresponding `parseElement` method in `app.forcefield.parser` to parse itself, and register `generator`.

`app.forcefield.parser` is a `dict`, the keys are Force tag names, and the values are `parseElement` method of `generator`. The Hamiltonian parse XML file will use the tag name to look up the corresponding `parseElement` method—the `generator` instance stores raw data from the XML file. You can use generators by `getGenerators()` in Hamiltonian. 

### Generator class

The generator class in charge of input file analysis, molecular topology construction, atomic classification, and expansion from force field parameter layer to atomic parameter layer. It is a middle layer link `Hamiltonian` and backend. See the following documents for the specific design logic:

![generator](../assets/generator.svg)

The custom generator must define those methods:


* @staticmethod parseElement(element, hamiltonian): OpenMM use `element.etree` parse tag in XML file, and `element` is `Element` object. For instance, if there were a section in XML file that defines bond potential:

```
  <HarmonicJaxBondForce>
    <Bond type1="ow" type2="hw" length="0.0957" k="462750.4"/>
    <Bond type1="hw" type2="hw" length="0.1513" k="462750.4"/>
  <\HarmonicJaxBondForce>
```

will activate `HarmonicBondJaxGenerator.parseElement` method which is the value of key `app.forcefield.parsers["HarmonicBondForce"]`. You can use `element.findall("Bond")` to get a iterator of the `Element` object of <Bond> tage. For `Element` object, you can use `.attrib` to get {'type1': 'ow} properties in `dict` format.

What `parseElement` does is parse the `Element` object and initialize the generator itself. The parameters in generators can be classified into two categories. Those differentiable should store in a `dict` named `dict`, and non-differentiable static parameters can simply be set as the generator's attribute. Jax support `pytree` nested container, and `params` can be directly grad.

* `createForce(self, system, data, nonbondedMethod, *args)` pre-process XML parameters, initialize calculator. `System` and `data` are given by OpenMM's forcefield class, which stores topology/atomType information(For now, you need to use debug-tool to access). To avoid breaking the differentiate chain, from XML raw data to per-atom properties, we should use `data` to directly construct per-atom info. Here is an example:

```
map_atomtype = np.zeros(n_atoms, dtype=int)

for i in range(n_atoms):
    atype = data.atomType[data.atoms[i]]
    map_atomtype[i] = np.where(self.types == atype)[0][0]
```

Finally, we need to bind the calculator's compute function to `self._jaxPotential`

```
def potential_fn(positions, box, pairs, params):
    return bforce.get_energy(
        positions, box, pairs, params["k"], params["length"]
    )

self._jaxPotential = potential_fn
```

All parameters accepted by `potential_fn` should be differentiable. Non-differentiable parameters are passed into it by closure(see code convention section). Meanwhile, if the generator needs to initialize multiple calculators(e.g., `NonBondedJaxGenerator` will call `LJ` and `PME` two kinds of calculators), `potential_fn` will return the summation of two potential energy. 

Here is a pseudo-code of the frontend module, demonstrating basic API and method

```
from openmm import app

class SimpleJAXGenerator:
    
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = None
        self._jaxPotential = None
        init_other_attributes_if_needed
        
    @staticmethod
    def parseElement(element, hamiltonian):
        parse_xml_element
        generator = SimpleGenerator(hamiltonian, args_from_xml)
        hamiltonian.registerGenerator(generator)
        
    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        generate_constraints_if_needed
        # Create JAX energy function from system information
        create_jax_potential
        self._jaxPotential = jaxPotential
        
    def getJaxPotential(self, data, **args):
        return self._jaxPotential
        
    def renderXML(self):
        render_xml_forcefield_from_params
        
        
app.parsers["SimpleJAXForce"] = SimpleJAXGenerator.parseElement

class Hamiltonian(app.ForceField):

    def __init__(self, **args):
        super(app.ForceField, self).__init__(self, **args)
        self._potentials = []
        
    def createPotential(self, topology, **args):
        system = self.createSystem(topology, **args)
        load_constraints_from_system_if_needed
        # create potentials
        for generator in self._generators:
            potentialImpl = generator.getJaxPotential(data)
            self._potentials.append(potentialImpl)
        return [p for p in self._potentials]
```

And here is a HarmonicBond potential implement:

```
class HarmonicBondJaxGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = {"k": [], "length": []}
        self._jaxPotential = None
        self.types = []

    def registerBondType(self, bond):

        types = self.ff._findAtomTypes(bond, 2)

        # self.ff._findAtomTypes是OpenMM预先实现的type/class匹配函数。
        # 第一个入参填入xml element，第二个入参填入需要匹配的type个数。
        # 返回值的格式为：
        #     [[atype1, atype2, ...], [atype3, atype4, ...], ...]
        # 对于按atomtype匹配的情况，这一函数会返回一个list，其中含有匹配到
        # 的atomtype；对于按class匹配的情况，这一函数会返回一个含有class
        # 中所有atomtype的list
        
        self.types.append(types)
        self.params["k"].append(float(bond["k"]))
        self.params["length"].append(float(bond["length"]))  # length := r0

    @staticmethod
    def parseElement(element, hamiltonian):

        # 处理xml tree，element为匹配到的力场参数结点。
        # 使用element.findall与element.attrib获取其子节点与节点内参数。

        generator = HarmonicBondJaxGenerator(hamiltonian)
        hamiltonian.registerGenerator(generator)
        for bondtype in element.findall("Bond"):
            generator.registerBondType(bondtype.attrib)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff, args):
        # jax it!
        # 将self.params内所有ndarray转为jax array
        for k in self.params.keys():
            self.params[k] = jnp.array(self.params[k])
        self.types = np.array(self.types)

        n_bonds = len(data.bonds)
        
        # data为OpenMM对体系拓扑进行处理后构造的数据文件。
        # 这一文件记录了所有的bonds、angles、dihedrals和impropers。
        # 同时也记录了每一个粒子的atomtype。
        # 调用方式如下：

        map_atom1 = np.zeros(n_bonds, dtype=int)
        map_atom2 = np.zeros(n_bonds, dtype=int)
        map_param = np.zeros(n_bonds, dtype=int)
        for i in range(n_bonds):
            idx1 = data.bonds[i].atom1
            idx2 = data.bonds[i].atom2
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            ifFound = False
            for ii in range(len(self.types)):
                if (type1 in self.types[ii][0] and type2 in self.types[ii][1]) or (
                    type1 in self.types[ii][1] and type2 in self.types[ii][0]
                ):
                    map_atom1[i] = idx1
                    map_atom2[i] = idx2
                    map_param[i] = ii
                    ifFound = True
                    break
            if not ifFound:
                raise BaseException("No parameter for bond %i - %i" % (idx1, idx2))
        
        # 外部引入的函数，用于创建jax energy function。
        # 在这一例子中，bond与parameter对应关系的indices列表
        # 在创建energy function时被预先传入。
        bforce = HarmonicBondJaxForce(map_atom1, map_atom2, map_param)

        # potential_fn 是用来调用energy function的函数。
        # 在这一函数中，参数字典中的各个键值被正确传进energy function中。
        def potential_fn(positions, box, pairs, params):
            return bforce.get_energy(
                positions, box, pairs, params["k"], params["length"]
            )

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

# register all parsers
app.forcefield.parsers["HarmonicBondForce"] = HarmonicBondJaxGenerator.parseElement
```

## How backend works

The so-called *backend* module is the bearer of the workload of calculating potential. This module is independent, so it has very flexiable form. For example, Here is a toy-code which compute Harmonic Bond potential:

```python
def distance(p1v, p2v):
    return jnp.sqrt(jnp.sum(jnp.power(p1v - p2v, 2), axis=1))
    

class HarmonicBondJaxForce:

    def __init__(self, p1idx, p2idx, prmidx):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.prmidx = prmidx
        self.refresh_calculators()

    def generate_get_energy(self):

        def get_energy(positions, box, pairs, k, length):
            p1 = positions[self.p1idx]
            p2 = positions[self.p2idx]
            kprm = k[self.prmidx]
            b0prm = length[self.prmidx]
            dist = distance(p1, p2)
            return jnp.sum(0.5 * kprm * jnp.power(dist - b0prm, 2))

        return get_energy

    def update_env(self, attr, val):
        """
        Update the environment of the calculator
        """
        setattr(self, attr, val)
        self.refresh_calculators()

    def refresh_calculators(self):
        """
        refresh the energy and force calculators according to the current environment
        """
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
```
