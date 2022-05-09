# Tutorial

## Write XML

DMFF uses a simple XML file to describe force fields. Let us take an example of writing a DMFF XML file using the classical force field to calculate the water molecule system.

Support we treat the water molecule as a three-body molecule. Within the molecule, we need harmonic interaction to describe the bonded interaction and harmonic angle potential. Between molecules, the interactions between atoms are expressed through the Lennard-jones potential.

Let us create a new file called `forcefield.xml`. The root element of the XML file must be a `<ForceField>` tag:

```
<ForceField>
...
</ForceField>
```

The `<ForceField>` tag contains the following children:

- An `<AtomTypes>` tag containing the atom type definitions

- A `<Residues>` tag containing the residue template definitions

- Zero or more tags defining specific forces

The order of these tags does not matter. They are described in detail below.

`<AtomTypes>` defines atom type in the System. In this case, we have two types of atom:


```
<AtomTypes>
    <Type element="O" name="oh" class="oh" mass="15.999" />
    <Type element="H" name="ho" class="ho" mass="1.008" />
</AtomTypes>
```

Each `<Type>` tag in this section represents a type of atom. It specifies the name of the type, the class it belongs to, the symbol for its element, and its mass. The names are arbitrary strings: they need not be numbers, as in this example. The only requirement is that all types have unique names. The classes are also arbitrary strings and in general will not be unique. If they list the same value for the class attribute, two types belong to the same class. 

The residue template definitions look like this:

```
<Residues>
    <Residue name="h2o" nametype="classical">
        <Atom name="O" type="oh" charge="-0.8476" mass="15.999" />
        <Atom name="H1" type="ho" charge="0.4238" mass="1.008" />
        <Atom name="H2" type="ho" charge="0.4238" mass="1.008" />
        <Bond atomName1="O" atomName2="H1" />
        <Bond atomName1="O" atomName2="H2" />
    </Residue>
</Residues>
```

`<Residues>` template contains the following tags:

- An `<Atom>` tag for each atom in the residue. This specifies the name of the atom and its atom type.

- A `<Bond>` tag for each pair of atoms that are bonded to each other. The atomName1 and atomName2 attributes are the names of the two bonded atoms. (Some older force fields use the alternate tags to and from to specify the atoms by index instead of name. This is still supported for backward compatibility, but specifying atoms by name is recommended since it makes the residue definition much easier to understand.)

The `<Residue>` tag may also contain `<VirtualSite>` tags, as in the following example:


```
    <Residue name="HOH">
    <Atom name="O" type="tip4pew-O"/>
    <Atom name="H1" type="tip4pew-H"/>
    <Atom name="H2" type="tip4pew-H"/>
    <Atom name="M" type="tip4pew-M"/>
    <VirtualSite type="average3" siteName="M" atomName1="O" atomName2="H1" atomName3="H2"
        weight1="0.786646558" weight2="0.106676721" weight3="0.106676721"/>
    <Bond atomName1="O" atomName2="H1"/>
    <Bond atomName1="O" atomName2="H2"/>
    </Residue>
```

Each `<VirtualSite>` tag indicates an atom in the residue that should be represented with a virtual site. The type attribute may equal "average2", "average3", "outOfPlane", or "localCoords", which correspond to the TwoParticleAverageSite, ThreeParticleAverageSite, OutOfPlaneSite, and LocalCoordinatesSite classes respectively. The siteName attribute gives the name of the atom to represent with a virtual site. The atoms it is calculated based on are specified by atomName1, atomName2, etc. (Some old force fields use the deprecated tags index, atom1, atom2, etc. to refer to them by index instead of name.)

The remaining attributes are specific to the virtual site class and specify the parameters for calculating the site position. For a TwoParticleAverageSite, they are weight1 and weight2. For a ThreeParticleAverageSite, they are weight1, weight2, and weight3. For an OutOfPlaneSite, they are weight12, weight13, and weightCross. For a LocalCoordinatesSite, they are p1, p2, and p3 (giving the x, y, and z coordinates of the site position in the local coordinate System), and wo1, wx1, wy1, wo2, wx2, wy2, … (giving the weights for computing the origin, x-axis, and y-axis).    

Next, to add a HarmonicBondForce to the System, include a tag that looks like this:

```
<HarmonicBondForce>
    <Bond type1="ho" type2="oh" length="0.0973" k="471536.79999999993"/>
</HarmonicBondForce>
```

Every `<Bond>` tag defines a rule for creating harmonic bond interactions between atoms. Each tag may identify the atoms either by type (using the attributes type1 and type2) or by class (using the attributes class1 and class2). For every pair of bonded atoms, the force field searches for a rule whose atom types or atom classes match the two atoms. If it finds one, it calls addBond() on the HarmonicBondForce with the specified parameters. Otherwise, it ignores that pair and continues. length is the equilibrium bond length in nm, and k is the spring constant in kJ/mol/nm2.

To add a HarmonicAngleForce to the System, include a tag that looks like this:

```
<HarmonicAngleForce>
    <Angle type1="ho" type2="oh" type3="ho" angle="1.7229890375688022" k="519.6528000000001"/>
</HarmonicAngleForce>
```

Every `<Angle>` tag defines a rule for creating harmonic angle interactions between triplets of atoms. Each tag may identify the atoms either by type (using the attributes type1, type2, …) or by class (using the attributes class1, class2, …). The force field identifies every set of three atoms in the System where the first is bonded to the second, and the second to the third. For each one, it searches for a rule whose atom types or atom classes match the three atoms. If it finds one, it calls addAngle() on the HarmonicAngleForce with the specified parameters. Otherwise, it ignores that set and continues. angle is the equilibrium angle in radians, and k is the spring constant in kJ/mol/radian2.

To add a NonbondedForce to the System, include a tag that looks like this:

```
<NonbondedForce coulomb14scale="0.83333333" lj14scale="0.5">
    <UseAttributeFromResidue name="charge"/>
    <Atom type="ho" sigma="0.053792464601313685" epsilon="0.0196648"/>  
    <Atom type="oh" sigma="0.3242871334030835" epsilon="0.389112"/>      
</NonbondedForce>
```

Each `<Atom>` tag specifies the OBC parameters for one atom type (specified with the type attribute) or atom class (specified with the class attribute). It is fine to mix these two methods, having some tags specify a type and others specify a class. However you do it, you must make sure that a unique set of parameters is defined for every atom type. charge is measured in units of the proton charge, radius is the GBSA radius in nm, and scale is the OBC scaling factor.

This is what we should do to describe a simple system with a classical force field.

## Write a run script

We already have a XML file to describe our System, now we need to write a python script to calculate energy and force. 

First, we need to parse PDB file

```
import openmm.app as app
pdb = app.PDBFile('/path/to/pdb')
positions = jnp.array(pdb.positions._value)
a, b, c = pdb.topology.getPeriodicBoxVectors()
box = jnp.array([a._value, b._value, c._value])
```

Second, a `Hamiltonian` class should be initialized with XML file path

```
from dmff.api import Hamiltonian
H = Hamiltonian('forcefield.xml')
rc = 4.0  # cutoff
system = H.createPotential(pdb.topology, nonbondedCutoff=rc)
```

The `Hamiltonian` class will parse tags in XML file and invoke corresponding potential functions. We can access those potentials in this way:

```
bondE = H._potentials[0]
angleE = H._potentials[1]
nonBondE = H._potentials[2]
```

> Note: only when the `createPotential` method is called can potentials be obtained

Next, we need to construct neighbor list. Here we use the code from `jax_md`:

```
from jax_md import space, partition
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
nbr = neighbor_list_fn.allocate(positions)
pairs = nbr.idx.T  
```

`pairs` is a `(N, 2)` shape array, which indicates the index of atom i and atom j. ATTENTION: pairs array contains many **invalid** index. For example, in this case, we only have 6 atoms and pairs' shape maybe `(18, 2)`. And even there are three `[6, 6]` pairs which are obviously out of range. Because `jax-md` takes advantage of the feature of Jax.numpy, which will not throw an error when the index out of range, and return the [last element](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing).

Finally, we can calculate energy and force using the aforementioned potential:

```
print("Bond:", value_and_grad(bondE)(positions, box, pairs, H.getGenerators()[0].params))
print("Angle:", value_and_grad(angleE)(positions, box, pairs, H.getGenerators()[1].params))
print('NonBonded:', value_and_grad(nonBondE)(positions, box, pairs, H.getGenerators()[2].params))    
```

also, we can write a simple gradient descent to optimize parameters:

```
import optax
# start to do optmization
lr = 0.001
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

n_epochs = 1000
for i_epoch in range(n_epochs):
        loss, grads = value_and_grad(bondE, argnums=(0))(params, data[sid])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    with open('params.pickle', 'wb') as ofile:
        pickle.dump(params, ofile)
```