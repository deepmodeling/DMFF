# 4. XML format force field

The design of openmm force field file is quite modular and convenient to use. Unfortunately, only limited documentations are available right now to explain the details of the file format. Here, the format and the meaning of the OpenMM XML file are sorted in details in below.

Overall speaking, the typification process is composed by the following steps:

1. Build the residue topology (draw all bonds) according to the residue name. This is done by matching the residue template in the topology file.

2. Match the residue topology with the right parameter template.

3. Use the information in the matched parameter template to define the potential function. More specifically, go over all forces (i.e., potential components), and for each force (e.g., the bond stretching potential), match all terms (e.g., find all bonds) and register these terms in the total potential. 

The files involved in this process are introduced below.

## Topology file

Topology file is used to describe the bonding information of residues. Whenever a residue name is matched, the OpenMM topology module will add keys for atoms according to the information in the topology file.

An example of the residue topology XML file is as follows:
```xml
<!-- residues.xml -->
<Residues>
    <Residue name="ALA">
        <Bond from="-C" to="N"/>
        <Bond from="C" to="CA"/>
        <Bond from="C" to="O"/>
        <Bond from="C" to="OXT"/>
        <Bond from="CA" to="CB"/>
        <Bond from="CA" to="HA"/>
        <Bond from="CA" to="N"/>
        <Bond from="CB" to="HB1"/>
        <Bond from="CB" to="HB2"/>
        <Bond from="CB" to="HB3"/>
        <Bond from="H" to="N"/>
        <Bond from="H2" to="N"/>
        <Bond from="H3" to="N"/>
        <Bond from="HXT" to="OXT"/>
    </Residue>
</Residues>
```

Where "-C" indicates an external bond between the "C" atom and the **previous** residue. During typification, the code will try to match all atoms in the topology template with the actual atoms in the real structure. Once a match is successful, the matched atom will be bonded accordingly. If the match fails, this template atom will be skipped. Therefore, the actual number of bonds in the matched structure can be less than the number of bonds defined in the template. 

The XML file registration method is as follows:

``` python
try:
    import openmm.app as app
except:
    import simtk.openmm.app as app
    
app.Topology.loadBondDefinations("residues.xml") # register residue topology

# Create topology and add atoms and residues to it, which is automatically performed when reading PDB
top = app.Topology()
...
top.createStandardBonds() # Connect keys according to template files
```

After this process, the bonding topologies are constructed in the matched residue, but the force field parameters are not yet assigned. It should be noted that disulfide bonds are not registered in this step. The OpenMM topology class will look for SG atoms in Cys that are not connected to Hg, and connect SG atom pairs within 0.3nm as disulfide bonds.

## Force field Parameter File

After all bonds are constructed using the topology file, the force field parameters will be assigned using the force field parameter file.

The force field parameter file is as follows:
``` xml
<!-- tip3p.xml -->
<ForceField>
    <Residues>
        <Residue name="HOH">
            <Atom name="O" type="spce-O" charge="-0.8476" />
            <Atom name="H1" type="spce-H" charge="0.4238" />
            <Atom name="H2" type="spce-H" charge="0.4238" />
            <Bond atomName1="O" atomName2="H1"/>
            <Bond atomName1="O" atomName2="H2"/>
        </Residue>
    </Residues>
    <AtomTypes>
        <Type name="spce-O" class="OW" element="O" mass="15.99943"/>
        <Type name="spce-H" class="HW" element="H" mass="1.007947"/>
    </AtomTypes>
    <HarmonicBondForce>
        <Bond class1="OW" class2="HW" length="0.1" k="462750.4"/>
    </HarmonicBondForce>
    <HarmonicAngleForce>
        <Angle class1="HW" class2="OW" class3="HW" angle="1.91061193216" k="836.8"/>
    </HarmonicAngleForce>
    <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
        <UseAttributeFromResidue name="charge"/>
        <Atom type="spce-O" sigma="0.31657195050398818" epsilon="0.6497752"/>
        <Atom type="spce-H" sigma="1" epsilon="0"/>
    </NonbondedForce>
</ForceField>
```
This file can be further divided into the residue part and the force field part.

### Residue Part
``` xml
<!-- tip3p.xml -->
<ForceField>
    <Residues>
        <Residue name="HOH">
            <Atom name="O" type="spce-O" charge="-0.8476" />
            <Atom name="H1" type="spce-H" charge="0.4238" />
            <Atom name="H2" type="spce-H" charge="0.4238" />
            <Bond atomName1="O" atomName2="H1"/>
            <Bond atomName1="O" atomName2="H2"/>
        </Residue>
    </Residues>
    ...
</ForceField>
```
The `<atom>` node of the residue part defines all the atoms involved in the residue and some paramemters per atom, which can be called by the force field part on demand. The `<bond>` node defines the bonding information of the residue. The information contained in this part is different from that in the topology file discussed above. Take ALA as an example, we usually have at least three states for ALA, N-end, C-end and in-chain. The corresponding parameter templates in the force field file are as follows:

``` xml
<Residue name="ALA">
  <Atom charge="-0.4157" name="N" type="protein-N"/>
  <Atom charge="0.2719" name="H" type="protein-H"/>
  <Atom charge="0.0337" name="CA" type="protein-CX"/>
  <Atom charge="0.0823" name="HA" type="protein-H1"/>
  <Atom charge="-0.1825" name="CB" type="protein-CT"/>
  <Atom charge="0.0603" name="HB1" type="protein-HC"/>
  <Atom charge="0.0603" name="HB2" type="protein-HC"/>
  <Atom charge="0.0603" name="HB3" type="protein-HC"/>
  <Atom charge="0.5973" name="C" type="protein-C"/>
  <Atom charge="-0.5679" name="O" type="protein-O"/>
  <Bond atomName1="N" atomName2="H"/>
  <Bond atomName1="N" atomName2="CA"/>
  <Bond atomName1="CA" atomName2="HA"/>
  <Bond atomName1="CA" atomName2="CB"/>
  <Bond atomName1="CA" atomName2="C"/>
  <Bond atomName1="CB" atomName2="HB1"/>
  <Bond atomName1="CB" atomName2="HB2"/>
  <Bond atomName1="CB" atomName2="HB3"/>
  <Bond atomName1="C" atomName2="O"/>
  <ExternalBond atomName="N"/>
  <ExternalBond atomName="C"/>
</Residue>
<Residue name="CALA">
  <Atom charge="-0.3821" name="N" type="protein-N"/>
  <Atom charge="0.2681" name="H" type="protein-H"/>
  <Atom charge="-0.1747" name="CA" type="protein-CX"/>
  <Atom charge="0.1067" name="HA" type="protein-H1"/>
  <Atom charge="-0.2093" name="CB" type="protein-CT"/>
  <Atom charge="0.0764" name="HB1" type="protein-HC"/>
  <Atom charge="0.0764" name="HB2" type="protein-HC"/>
  <Atom charge="0.0764" name="HB3" type="protein-HC"/>
  <Atom charge="0.7731" name="C" type="protein-C"/>
  <Atom charge="-0.8055" name="O" type="protein-O2"/>
  <Atom charge="-0.8055" name="OXT" type="protein-O2"/>
  <Bond atomName1="N" atomName2="H"/>
  <Bond atomName1="N" atomName2="CA"/>
  <\displaylines{Bond atomName1="CA" atomName2="HA"/>
  <Bond atomName1="CA" atomName2="CB"/>
  <Bond atomName1="CA" atomName2="C"/>
  <Bond atomName1="CB" atomName2="HB1"/>
  <Bond atomName1="CB" atomName2="HB2"/>
  <Bond atomName1="CB" atomName2="HB3"/>
  <Bond atomName1="C" atomName2="O"/>
  <Bond atomName1="C" atomName2="OXT"/>
  <ExternalBond atomName="N"/>
</Residue>
<Residue name="NALA">
  <Atom charge="0.1414" name="N" type="protein-N3"/>
  <Atom charge="0.1997" name="H1" type="protein-H"/>
  <Atom charge="0.1997" name="H2" type="protein-H"/>
  <Atom charge="0.1997" name="H3" type="protein-H"/>
  <Atom charge="0.0962" name="CA" type="protein-CX"/>
  <Atom charge="0.0889" name="HA" type="protein-HP"/>
  <Atom charge="-0.0597" name="CB" type="protein-CT"/>
  <Atom charge="0.03" name="HB1" type="protein-HC"/>
  <Atom charge="0.03" name="HB2" type="protein-HC"/>
  <Atom charge="0.03" name="HB3" type="protein-HC"/>
  <Atom charge="0.6163" name="C" type="protein-C"/>
  <Atom charge="-0.5722" name="O" type="protein-O"/>
  <Bond atomName1="N" atomName2="H1"/>
  <Bond atomName1="N" atomName2="H2"/>
  <Bond atomName1="N" atomName2="H3"/>
  <Bond atomName1="N" atomName2="CA"/>
  <Bond atomName1="CA" atomName2="HA"/>
  <Bond atomName1="CA" atomName2="CB"/>
  <Bond atomName1="CA" atomName2="C"/>
  <Bond atomName1="CB" atomName2="HB1"/>
  <Bond atomName1="CB" atomName2="HB2"/>
  <Bond atomName1="CB" atomName2="HB3"/>
  <Bond atomName1="C" atomName2="O"/>
  <ExternalBond atomName="C"/>
</Residue>
```

In this example, the atom numbers and the bonding configurations of ALA, CALA and NALA are different. When matching each ALA, OpenMM will try to match CALA, NALA, and ALA separately. It will compare each parameter template with the topology of the residue, and select the one with the right number of atoms, element composition, and bonding configurations as the matched template. The parameter template contains atom type and class information, which are then used to assign force field parameters.


### Forcefield Part
```xml
<!-- tip3p.xml -->
<ForceField>
    ...
    <AtomTypes>
        <Type name="spce-O" class="OW" element="O" mass="15.99943"/>
        <Type name="spce-H" class="HW" element="H" mass="1.007947"/>
    </AtomTypes>
    <HarmonicBondForce>
        <Bond class1="OW" class2="HW" length="0.1" k="462750.4"/>
    </HarmonicBondForce>
    <HarmonicAngleForce>
        <Angle class1="HW" class2="OW" class3="HW" angle="1.91061193216" k="836.8"/>
    </HarmonicAngleForce>
    <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
        <UseAttributeFromResidue name="charge"/>
        <Atom type="spce-O" sigma="0.31657195050398818" epsilon="0.6497752"/>
        <Atom type="spce-H" sigma="1" epsilon="0"/>
    </NonbondedForce>
</ForceField>
```

The `<atomtypes>` node defines all atom types. The `type` label of each atom in the residue part will match the `name` label of each child node of `<atomtypes>`. For each atom type, it also defines a `class` tag for different matching scenarios. The `name` tag of different `<type>` must be different, but the `class` tag can be the same.

The `<*Force>` node defines the matching rule of a potential function. For example, `<HarmonicBondForce>` defines harmonic bond, and the `<NonBondedForce>` node defines intermolecular interaction. For more information about each force, the readers are referred to this document: [details](http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html#writing-the-xml-file).

In the matching process, OpenMM will iterate all atoms, bonds, angles, dihedrals and impropers, and add all matched entries to the total potential function. Matching can be carried out according to the `type` tag, corresponding to the `name` of each atom defined in `<atomtype>`; It can also be based on the `class` tag, corresponding to the `class` of each atom in `<atomtype>`. This design is applicable to the situation when there are many types of atoms but they are roughly the same. For example, there are few kinds of LJ parameters in small molecular force field, but there are many kinds of intramolecular force parameters. We can even create a separate type for a specific small molecule to define the intra molecular interaction, but it belongs to the same class on LJ, so as to achieve the effect that the small molecule parameters can be tuned and do not affect each other.
