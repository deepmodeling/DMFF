# Architecture of DMFF

![arch](../assets/arch.png)

The overall framework of DMFF can be divided into two parts: parser & typing and calculators. We usually refer to the former as the *frontend* and the latter as the *backend* for ease of description.

DMFF introduces different forms of force fields in a modular way. For any form of force field, it is divided into frontend module and backend module. The frontend module is responsible for input file parsing, molecular topology construction, atomic typification, and unfolding from force field parameter layer to atomic parameter layer; The backend module is the calculation core, which calculates the energy & force of the system at a certain time by particle positions and system properties.

In the design of the front-end module, DMFF reuses the frontend parser module from OpenMM and realizes the functions of topology analysis. All frontend modules are stored in `api.py` and called by the Hamiltonian class.

The backend module is usually an automatically differentiable computing module built with Jax. 

In the following documents, the structure of front-end and back-end modules will be introduced in detail.

## How Frontend Works

Frontend modules are stored in api.py . `Hamiltonian` class is the top-level class exposed to users by DMFF. `Hamiltonian` class reads the path of the XML file, parses the XML file, and calls different frontend module according to the XML tags. The frontend module has same form with OpenMM's generator in its [forcefield.py](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py). The `Generator` class takes XML tag in and parse the parameters, initialize the backend calculator and provide interface of energy calculation method.

When users use the DMFF, the only thing need to do is initilize the the `Hamiltonian` class. In this process, `Hmiltonian` will automatically parse and initialize the corresponding potential function according to the tags in XML. The call logic is shown in the following chart. The box represents the command executed in Python script, and the rounded box represents the internal operation logic of OpenMM when executing the command.

![openmm_workflow](../assets/opemm_workflow.svg)

### Hamiltonian Class

Hamiltonian class is the top-level frontend module, which inherits from the [forcefield class](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py) of OpenMM. It is responsible for parsing XML force field files and generating potential functions to calculate system energy for given topology information. First, the usage of Hamiltonian class is given:


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

Hamiltonian class performs the following operations during instantiation:

* read Residue tag in XML, generate Residue template;
* read AtomTypes tag, store AtomType of each atom;
* for each Force tag, call corresponding `parseElement` method in `app.forcefield.parser` to parse itself, and register `generator`.

`app.forcefield.parser` is a `dict`, the keys are Force tag names and the values are `parseElement` method of `generator`. When Hamiltonian parse XML file, it will use tag name to look up corresponding `parseElement` method. The `generator` instance stores raw data from XML file, you can generators by `getGenerators()` in Hamiltonian. 

### Generator Class


The generator class in charge of input file analysis, molecular topology construction, atomic classification, and expansion from force field parameter layer to atomic parameter layer. It is a middle layer link `Hamiltonian` and backend. See the following documents for the specific design logic:

![generator](../assets/generator.svg)

The custom generator must define those methods:


* @staticmethod parseElement(element, hamiltonian): OpenMM use `element.etree` parse tag in XML file, and `element` is `Element` object. For instance, if there were a section in XML file that defines bond potential:

```xml
  <HarmonicJaxBondForce>
    <Bond type1="ow" type2="hw" length="0.0957" k="462750.4"/>
    <Bond type1="hw" type2="hw" length="0.1513" k="462750.4"/>
  </HarmonicJaxBondForce>
```

will activate `HarmonicBondJaxGenerator.parseElement` method which is the value of key `app.forcefield.parsers["HarmonicBondForce"]`. You can use `element.findall("Bond")` to get a iterator of the `Element` object of <Bond> tage. For `Element` object, you can use `.attrib` to get {'type1': 'ow} properties in `dict` format.

What `parseElement` do is parse `Element` object and initialize generator itself. The parameters in generators can be classified into to category. Those differentiable shoud store in a `dict` named `dict`, and non-differentiable static parameters can simply set as generator's attribute. Jax support pytree nested container, and `params` can be directly grad.

  
* `createForce(self, system, data, nonbondedMethod, *args)` pre-process XML parameters, initialize calculator. `System` and `data` are given by OpenMM's forcefield class, which store topology/atomType information (For now you need to use debug tool to access). To avoid break the differentiate chain, from XML raw data to per-atom properties, we should use `data` to construct per-atom info directly. Here is an example:

```python
map_atomtype = np.zeros(n_atoms, dtype=int)

for i in range(n_atoms):
    atype = data.atomType[data.atoms[i]]
    map_atomtype[i] = np.where(self.types == atype)[0][0]
```

Finally, we need to bind calculator's compute function to `self._jaxPotential`


```python
            
def potential_fn(positions, box, pairs, params):
    return bforce.get_energy(
        positions, box, pairs, params["k"], params["length"]
    )

self._jaxPotential = potential_fn
```

All parameters accepted by `potential_fn` should be differentiable. Non differentiable parameters are passed into it by closure (see code convention section). Meanwhile, if the generator need to initialize multiple calculators (e.g. `NonBondedJaxGenerator` will call `LJ` and `PME` two kinds of calculators), `potential_fn` will return the summation of two potential energy. 

Here is a pesudo code of frontend module, demostrate basic api and method

```python
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

```python
class HarmonicBondJaxGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = {"k": [], "length": []}
        self._jaxPotential = None
        self.types = []

    def registerBondType(self, bond):

        types = self.ff._findAtomTypes(bond, 2)
        # self.ff._findAtomTypes is a function implemented in OpenMM to patch 
        # atom types. The first argument is xml element. The second argument is 
        # the number of types needed to be patched. 
        # The return of this function is:
        # [[atype1, atype2, ...], [atype3, atype4, ...], ...]
        # When patching by atom types, the function would return a list with 
        # patched atom types. When patching by atom classes, the function would 
        # return a list with all the atom types patched to the class.
        
        self.types.append(types)
        self.params["k"].append(float(bond["k"]))
        self.params["length"].append(float(bond["length"]))  # length := r0

    @staticmethod
    def parseElement(element, hamiltonian):
        # Work with xml tree. Element is the node of forcefield.
        # Use element.findall and element.attrib to get the 
        # children nodes and attributes in the node.
            
        generator = HarmonicBondJaxGenerator(hamiltonian)
        hamiltonian.registerGenerator(generator)
        for bondtype in element.findall("Bond"):
            generator.registerBondType(bondtype.attrib)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff, args):
        # jax it!
        for k in self.params.keys():
            self.params[k] = jnp.array(self.params[k])
        self.types = np.array(self.types)

        n_bonds = len(data.bonds)
        # data is the data structure built by OpenMM, saving topology information of the system.
        # The object maintains all the bonds, angles, dihedrals and impropers.
        # And it also maintains the atomtype of each particle.
        # Use data.atoms, data.bonds, data.angles, data.dihedrals, data.impropers 
        # to get the atom types.
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
        
        # HarmonicBondJaxForce is the backend class to build potential function
        bforce = HarmonicBondJaxForce(map_atom1, map_atom2, map_param)

        # potential_fn is the function to call potential, in which the dict self.params
        # is fed to harmonic bond potential function
            
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

## How Backend Works

### Force Class

Force class is the module to build potential function. It does not require OpenMM and can 
be very flexible. For instance, the Force class of harmonic bond potential is shown below 
as an example of jax potential function.

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
