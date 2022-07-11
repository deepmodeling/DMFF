# 2. Software architecture

![arch](../assets/arch.png)

The overall architechture of DMFF can be divided into two parts: 1. parser & typing and 2. calculators. 
We usually refer to the former as the *frontend* and the latter as the *backend* for ease of description.

DMFF introduces different forms of force fields in a modular way. For each force field form 
(in OpenMM, each form is also called a `Force`), there is a frontend module and a backend module. 
The frontend module is responsible for input file parsing, molecular topology construction, atom typification, 
and dispatching the forcefield parameters into the atomic parameter; The backend module is the calculation kernel, 
which calculates the energy & force of the system, using particle positions and atomic parameters as inputs.

In the design of the frontend modules, DMFF reuses the frontend parser from OpenMM for topology analysis. 
The core class in frontend is the `Generator` class, which should be defined for each force field form. 
All frontend `Generators` are currently put in `api.py` and are called by the top-level `Hamiltonian` class.

The backend module is usually an automatically differentiable calculator built with Jax. 

The structures of the frontend and the backend modules will be introduced in detail in below.

## 2.1 Frontend

> NOTE: The front-end API has been re implemented after version 0.1.2 and is no longer dependent on openmm. It completely incompatible with previous versions. Please refactor your generator according to the instruction below.

Frontend modules are stored in `api.py`. `Hamiltonian` class is the top-level class exposed to users by DMFF. 
`Hamiltonian` class reads the path of the XML file, parses the XML file, and calls different frontend modules according to the `*Force` tags found in XML. It also provides a convenient interface to let all the generators extract, overwrite, render parameters.

When users use DMFF, the only thing they need to do is to initilize the the `Hamiltonian` class. 
In this process, `Hamiltonian` will automatically parse and initialize the corresponding potential function according to the tags in XML. 

### 2.1.1 Hamiltonian Class

The `Hamiltonian` class is the top-level frontend module, which inherits the 
[forcefield class](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py) in OpenMM. 
It is responsible for parsing the XML force field files and generating potential functions to calculate system energy 
with the given topology information. First, the usage of the `Hamiltonian` class is given:


```python
H = Hamiltonian('forcefield.xml')
app.Topology.loadBondDefinitions("residues.xml")
pdb = app.PDBFile("waterbox_31ang.pdb")
rc = 4.0
# generator stores all force field parameters
generators = H.getGenerators()
disp_generator = generators[0]
pme_generator = generators[1]

potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
# pot_fn is the actual energy calculator
pot_disp = potentials[0]
pot_pme = potentials[1]
```

`Hamiltonian` class performs the following operations during instantiation:

* read Residue tag in XML, generate Residue template;
* read AtomTypes tag, store AtomType of each atom;
* for each Force tag, call corresponding `parseElement` method in `app.forcefield.parser` to parse itself, and register `generator`.

`app.forcefield.parser` is a `dict`, the keys are Force tag names, and the values are the `parseElement` method 
of the corresponding `generator`. When `Hamiltonian` parses the XML file, it will use the tag name to look up the 
corresponding `parseElement` method, then calls it to initialize the `generator` instance, which stores the raw 
parameters from the XML file. You can access all the generators by the `getGenerators()` method in Hamiltonian. 


### 2.1.2 Generator Class


The generator class is in charge of input file analysis, molecular topology construction, atom classification, 
and expanding force field parameters to atomic parameters. It is a middle layerthat links `Hamiltonian` with the backend. Here is an example of the `HarmonicBondForce` generator, you can find it in the `api.py`.

In many cases, the parameters of the force fields are interdependent, so it is not enough for each generator to store only the parameters it needs. In the new version of frontend api, the constructor of a generator needs to include the following four members:

```xml
<HarmonicBondForce>
   <Bond type1="ow" type2="hw" length="0.09572000000000001" k="462750.3999999999"/>
<\HarmonicBondForce>
```

```python
class HarmonicBondJaxGenerator:
    def __init__(self, ff:Hamiltonian):
        self.name = "HarmonicBondForce"
        self.ff:Hamiltonian = ff
        self.fftree:ForcefieldTree = ff.fftree
        self.paramtree:Dict = ff.paramtree
```

The name must correspond to the tag to be parsed in the XML file. In the snippet, the name of  `HarmonicBondJaxGenerator` is `HarmonicBondForce` to parse the tag in the XML file and store those parameters. The `ff` is the `Hamiltonian` class, which is the top-level class of DMFF and manages all the generators. The `fftree` is the `ForcefieldTree` class, which is the tree-like storage of ALL force field parameters. The `paramtree` is the `Dict` class, and it stores differentiable parameters of this generator. 

The generator also should implement the dual methods, `extract` and `overwrite`. `extract` method extracts the paramters needed by the generator from the `fftree` and `overwrite` method overwrites(updates) the `fftree` by using the parameters in the `paramtree`. This process will be completed using the API of `ForcefieldTree`, which will be described later. In this example, we need to get `length` and `k` from XML file and update `fftree`:

```python
    def extract(self):
        """
        extract forcefield paramters from ForcefieldTree. 
        """
        lengths = self.fftree.get_attribs(f"{self.name}/Bond", "length")
        ks = self.fftree.get_attribs(f"{self.name}/Bond", "k")
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["length"] = jnp.array(lengths)
        self.paramtree[self.name]["k"] = jnp.array(ks)

    def overwrite(self):
        """
        update parameters in the fftree by using paramtree of this generator.
        """
        self.fftree.set_attrib(f"{self.name}/Bond", "length",
                               self.paramtree[self.name]["length"])
        self.fftree.set_attrib(f"{self.name}/Bond", "k",
                               self.paramtree[self.name]["k"])
```
In this case, you can use the `get_attribs` method to get the parameters from the `ForcefieldTree`, it will return a list of parameters, the omitted parameters will be filled with `None`. You shoud use `jnp.array` to convert it to a jax numpy array after handling `None` elements appropriately.
The parameters extract from `fftree` can be classified into two categories. The first category is differentiable parameters 
(such as `length` and `k`), which may be subject to further optimization. By design, these parameters should be fed into the potential 
function as explicit input arguments. Therefore, these parameters should be gathered in a `dict` object named `params`, which is then 
saved as an attribute of the `generator`. The second category is non-differentiable variables (such as `type1` and `type2`): it is 
unlikely that you are going to optimize them, so they are also called *static* variables. These variables will be used by the potential 
function implicitly and not be exposed to the users. Therefore, you may save them in `generator` at will, as long as they can be 
accessed by the potential function later.

After the calculation and optimization, we need to save the optimized parameters as XML format files for the next calculation. This serialization process is implemented through the `overwrite` method. In the `ForcefieldTree` class at the `fftree.py`, we have implemented the `set_attrib` method to set the attribute of the XML node. 

All the parameters needed by the generator are stored in the `paramtree`, and `createForce` method should be implemented to use the parameters to initialize the backend calculator. 
`createForce(self, system, data, nonbondedMethod, *args)` pre-process the force field parameters from XML, and use them to initialize
the backend calculators, then wrap the calculators using a potential function and returns it. 
`System` and `data` are given by OpenMM's forcefield class, which store topology/atomType information (for now you need to use 
debug tool to access). Bear in mind that we should not break the derivative chain from the XML raw data (force field parameters) 
to the per-atom properties (atomic parameters). So we should do the parameter dispatch (i.e., dispatching force field parameters to 
each atom) within the returned potential function. Therefore, in `createForce`, we usually only construct the atomtype index map using 
the information in `data`, but do not dispatch parameters! The atomtype map will be used in potential function implicitly, to dispatch
parameters.
It also provides the interface to wrap the backend calculators into potential functions, which are eventually returned to the users. 

Here is an example:

```python
map_atomtype = np.zeros(n_atoms, dtype=int)

for i in range(n_atoms):
    atype = data.atomType[data.atoms[i]]
    map_atomtype[i] = np.where(self.types == atype)[0][0]
```

Finally, we need to bind the calculator's compute function to `self._jaxPotential`, which is the final potential function (`potential_fn`) 
returned to users:

```python
            
def potential_fn(positions, box, pairs, params):
    isinstance_jnp(positions, box, params)
    return bforce.get_energy(
        positions, box, pairs, params["k"], params["length"]
    )

self._jaxPotential = potential_fn
```

The `potential_fn` function only takes `(positions, box, pairs, params)` as explicit input arguments. All these arguments except
`pairs` (neighbor list) should be differentiable. A helper function `isinstance_jnp` in `utils.py` can check take-in arguments whether they are `jnp.array`. Non differentiable parameters are passed into it by closure (see code convention section). 
Meanwhile, if the generator needs to initialize multiple calculators (e.g. `NonBondedJaxGenerator` will call both `LJ` and `PME` calculators), 
`potential_fn` should return the summation of the results of all calculators. 

Here is a pseudo-code of the frontend module, demonstrating basic API and method

```python

class SimpleJAXGenerator:
    
    def __init__(self, ff:Hamiltonian):
        self.name = "SimpleJAXGenerator"
        self.ff:Hamiltonian = ff
        self.fftree:ForcefieldTree = ff.fftree
        self.paramtree:Dict = ff.paramtree
        
    def extract(self):
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["a"] = jnp.array(self.fftree.get_attrib(f"{self.name}/tag", "a"))
        self.paramtree[self.name]["b"] = jnp.array(self.fftree.get_attrib(f"{self.name}/tag", "b"))

    def overwrite(self):

        self.fftree.set_attrib(f"{self.name}/tag", "a", self.paramtree[self.name]["a"])
        self.fftree.set_attrib(f"{self.name}/tag", "b", self.paramtree[self.name]["b"])
        
    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        generate_constraints_if_needed
        # Create JAX energy function from system information
        create_jax_potential
        self._jaxPotential = jaxPotential
        
    def getJaxPotential(self, data, **args):
        return self._jaxPotential       
        
jaxGenerators["HarmonicBondForce"] = HarmonicBondJaxGenerator

class Hamiltonian(app.ForceField):

    def __init__(self, **args):
        super(app.ForceField, self).__init__(self, **args)
        self._potentials = []
        
    def createPotential(self, topology, **args):
        system = self.createSystem(topology, **args)
        load_constraints_from_system_if_needed
        # create potentials
        potObj = Potential()
        potObj.addOmmSystem(system)
        for generator in self._jaxGenerators:
            if len(jaxForces) > 0 and generator.name not in jaxForces:
                continue
            try:
                potentialImpl = generator.getJaxPotential()
                potObj.addDmffPotential(generator.name, potentialImpl)
            except Exception as e:
                print(e)
                pass

        return potObj
```

And here is a HarmonicBond potential implement:

```python
class HarmonicBondJaxGenerator:
    def __init__(self, ff:Hamiltonian):
        self.name = "HarmonicBondForce"
        self.ff:Hamiltonian = ff
        self.fftree:ForcefieldTree = ff.fftree
        self.paramtree:Dict = ff.paramtree

    def extract(self):
        """
        extract forcefield paramters from ForcefieldTree. 
        """
        lengths = self.fftree.get_attribs(f"{self.name}/Bond", "length")
        # get_attribs will return a list if attribute name is a simple string
        # e.g. if you provide 'length' then return  [1.0, 2.0, 3.0];
        # if attribute name is a list, e.g. ['length', 'k']
        # then return [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]] List[List[value1, value2]]
        ks = self.fftree.get_attribs(f"{self.name}/Bond", "k")
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["length"] = jnp.array(lengths)
        self.paramtree[self.name]["k"] = jnp.array(ks)

    def overwrite(self):
        """
        update parameters in the fftree by using paramtree of this generator.
        """
        self.fftree.set_attrib(f"{self.name}/Bond", "length",
                               self.paramtree[self.name]["length"])
        self.fftree.set_attrib(f"{self.name}/Bond", "k",
                               self.paramtree[self.name]["k"])

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        This method will create a potential calculation kernel. It usually should do the following:
        
        1. Match the corresponding bond parameters according to the atomic types at both ends of each bond.

        2. Create a potential calculation kernel, and pass those mapped parameters to the kernel.

        3. assign the jax potential to the _jaxPotential.

        Args:
            Those args are the same as those in createSystem.
        """

        # initialize typemap
        matcher = TypeMatcher(self.fftree, "HarmonicBondForce/Bond")

        map_atom1, map_atom2, map_param = [], [], []
        n_bonds = len(data.bonds)
        # build map
        for i in range(n_bonds):
            idx1 = data.bonds[i].atom1
            idx2 = data.bonds[i].atom2
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            ifFound, ifForward, nfunc = matcher.matchGeneral([type1, type2])
            if not ifFound:
                raise BaseException(
                    f"No parameter for bond ({idx1},{type1}) - ({idx2},{type2})"
                )
            map_atom1.append(idx1)
            map_atom2.append(idx2)
            map_param.append(nfunc)
        map_atom1 = np.array(map_atom1, dtype=int)
        map_atom2 = np.array(map_atom2, dtype=int)
        map_param = np.array(map_param, dtype=int)

        bforce = HarmonicBondJaxForce(map_atom1, map_atom2, map_param)

        def potential_fn(positions, box, pairs, params):
            return bforce.get_energy(positions, box, pairs,
                                     params[self.name]["k"],
                                     params[self.name]["length"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


jaxGenerators["HarmonicBondForce"] = HarmonicBondJaxGenerator
```    

### 2.1.3 fftree

We organize the forcefield parameters in a ForcefieldTree. This class provides four method to read and write the parameters in XML file. Once you need to develop a new generator in DMFF, the only way you access and modify the parameters is through the ForcefieldTree.

```python
class ForcefieldTree(Node):
    def __init__(self, tag, **attrs):

        super().__init__(tag, **attrs)

    def get_nodes(self, parser:str)->List[Node]:
        """
        get all nodes of a certain path

        Examples
        --------
        >>> fftree.get_nodes('HarmonicBondForce/Bond')
        >>> [<Bond 1>, <Bond 2>, ...]

        Parameters
        ----------
        parser : str
            a path to locate nodes

        Returns
        -------
        List[Node]
            a list of Node
        """
        steps = parser.split("/")
        val = self
        for nstep, step in enumerate(steps):
            name, index = step, -1
            if "[" in step:
                name, index = step.split("[")
                index = int(index[:-1])
            val = [c for c in val.children if c.tag == name]
            if index >= 0:
                val = val[index]
            elif nstep < len(steps) - 1:
                val = val[0]
        return val

    def get_attribs(self, parser:str, attrname:Union[str, List[str]])->List[Union[value, List[value]]]:
        """
        get all values of attributes of nodes which nodes matching certain path

        Examples:
        ---------
        >>> fftree.get_attribs('HarmonicBondForce/Bond', 'k')
        >>> [2.0, 2.0, 2.0, 2.0, 2.0, ...]
        >>> fftree.get_attribs('HarmonicBondForce/Bond', ['k', 'r0'])
        >>> [[2.0, 1.53], [2.0, 1.53], ...]

        Parameters
        ----------
        parser : str
            a path to locate nodes
        attrname : _type_
            attribute name or a list of attribute names of a node

        Returns
        -------
        List[Union[float, str]]
            a list of values of attributes
        """
        sel = self.get_nodes(parser)
        if isinstance(attrname, list):
            ret = []
            for item in sel:
                vals = [convertStr2Float(item.attrs[an]) if an in item.attrs else None for an in attrname]
                ret.append(vals)
            return ret
        else:
            attrs = [convertStr2Float(n.attrs[attrname]) if attrname in n.attrs else None for n in sel]
            return attrs

    def set_node(self, parser:str, values:List[Dict[str, value]])->None:
        """
        set attributes of nodes which nodes matching certain path

        Parameters
        ----------
        parser : str
            path to locate nodes
        values : List[Dict[str, value]]
            a list of Dict[str, value], where value is any type can be convert to str of a number.

        Examples
        --------
        >>> fftree.set_node('HarmonicBondForce/Bond', 
                            [{'k': 2.0, 'r0': 1.53}, 
                             {'k': 2.0, 'r0': 1.53}])
        """
        nodes = self.get_nodes(parser)
        for nit in range(len(values)):
            for key in values[nit]:
                nodes[nit].attrs[key] = f"{values[nit][key]}"

    def set_attrib(self, parser:str, attrname:str, values:Union[value, List[value]])->None:
        """
        set ONE Attribute of nodes which nodes matching certain path

        Parameters
        ----------
        parser : str
            path to locate nodes
        attrname : str
            attribute name
        values : Union[float, str, List[float, str]]
            attribute value or a list of attribute values of a node

        Examples
        --------
        >>> fftree.set_attrib('HarmonicBondForce/Bond', 'k', 2.0)
        >>> fftree.set_attrib('HarmonicBondForce/Bond', 'k', [2.0, 2.0, 2.0, 2.0, 2.0])

        """
        if len(values) == 0:
            valdicts = [{attrname: values}]
        else:
            valdicts = [{attrname: i} for i in values]
        self.set_node(parser, valdicts)
```

The above source code gives clear type comments on how to use `fftree`. 

## 2.2 Backend

### 2.2.1 Force Class

Force class is the backend module that wraps the calculator function. 
It does not rely on OpenMM and can be very flexible. For instance, 
the Force class of harmonic bond potential is shown below as an example.

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

            # NOTE: pairs array from jax-md has invalid index
            pairs = regularize_pairs(pairs)  
            buffer_scales = pair_buffer_scales(pairs)            

            p1 = positions[self.p1idx]
            p2 = positions[self.p2idx]
            kprm = k[self.prmidx]
            b0prm = length[self.prmidx]
            dist = distance(p1, p2)
            return jnp.sum(0.5 * kprm * jnp.power(dist - b0prm, 2) * buffer_scales)  # mask invalid pairs

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

The design logic for the `Force` class is: it saves the *static* variables inside the class as 
the *environment* of the real calculators. Examples of the static environment variables include:
the $\kappa$ and $K_{max}$ in PME calculators, the covalent_map in real-space calculators etc.
For a typical `Force` class,  one needs to define the following methods:

* `__init__`: The initializer, saves all the *static* variables.
* `update_env(self, attr, val)`: updates the saved *static* variables, and refresh the calculators
* `generate_get_energy(self)`: generate a potential calculator named `get_energy` using the current 
  environment.
* `refresh_calculators(self)`: refresh all calculators if environment is updated.
  
In ADMP, all backend calculators only take atomic parameters as input, so they can be invoked
independently in hybrid ML/force field models. The dispatch of force field parameters is done 
in the `potential_fn` function defined in the frontend. 

Please note that the `pairs` array accepted by `get_energy` potential compute kernel is **directly** construct from `jax-md`'s neighborList. 
To keep the shape of array neat and tidy, prevent JIT the code every time `get_genergy` is called, the `pairs` array is padded. It has 
some invalid index in the padding area, say, those `pair_index==len(positions)` is invalid padding pair index. That means there are many 
`[len(positions), len(positions)]` pairs in the `pairs` array, resulting in the distance equl to 0. The solution is we first call `regularize_pairs` 
helper function to replace `[len(positions), len(positions)]` with `[len(positions)-2, len(positions)-1]`, so the distance is always non-zeros. Due 
to we compute additional invalid pairs, we need to compute a `buffer_scales` to mask out those invalid pairs. We need to use `pair_buffer_scales(pairs)` 
to get the mask, and apply it in the pair energy array before we sum it up. 
