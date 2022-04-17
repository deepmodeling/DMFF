# How to write XML file

The force field file design of openmm is quite modular and has high convenience. Unfortunately, there are few existing introductions and the documents are not clear enough. Now the format and meaning of OpenMM XML file are sorted as follows.

## Topology file

Topology file is used to describe the bonding information of residues. For molecules with residue name matching, the topology module of openmm will add keys for atoms according to the information in the XML file.

Examples of XML files are as follows:
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

Where "- C" indicates the connection with the "C" atom in the **previous** residue. During typification, all atoms in the residue with the same name will try to match. Once the typification is successful, it will be bonded, and if the matching fails, it will be skipped. Therefore, the actual number of bonds can be less than the number set in the template.

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

It should be noted that disulfide bond is not completed in this step. The OpenMM topology class will look for SG atoms in Cys that are not connected to Hg, and connect atom pairs less than 0.3nm as disulfide bonds.

## Force field parameter file

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
This document can be divided into residue part and force field part.

### residue part
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
The `<atom>` node of the residue part defines the atomtype of each atom in the residue and some parameter information of per atom, which can be called by the force field part on demand. The `<bond>` node defines the bonding information of residues. The information contained in this part is different from that in the topology file above. Take ALA as an example. For ALA, we usually need to define at least three state, N-end, C-end and in-chain. The template in the force field is as follows:

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
  <Bond atomName1="CA" atomName2="HA"/>
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

In this example, the atom number and bonding relationship of ALA, CALA and NALA are different. When matching each ALA, OpenMM will try to match CALA, NALA and ALA, and finally select the template with the same number of atoms, element composition and bonding relationship as the residue to define the force field parameters for each atom.

### forcefield part
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

The `<atomtypes>` node defines many atomic types. The `type` label of each atom in the residue part will match the `name` label of each child node of `<atomtypes>`. For each atom type, it also defines a `class` tag for different matching scenarios. The `name` of different `<type>` child nodes must be different, but the `class` can be the same.

The `<*force>` node defines the matching rule of a potential function. For example, `<HarmonicBondForce>` defines harmonic bond, and the `<NonBondedForce>` node defines intermolecular interaction. You can view the document for specific parameter [details](http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html#writing-the-xml-file)

In the matching process, OpenMM will iterate all atom, bond, angle, dihedral and improver, and add all matching entries to the total potential function. Matching can be carried out according to the `type` tag, corresponding to the `name` of each atom in `<atomtype>`; It can also be based on the `class` tag, corresponding to the `class` of each atom in `<atomtype>`. This design is applicable to the situation that there are many types of atoms but they are roughly the same. For example, there are few kinds of LJ parameters in small molecular force field, but there are many kinds of intramolecular force parameters. We can even create a separate type for a specific small molecule to define the intra molecular interaction, but it belongs to the same class on LJ, so as to achieve the effect that the small molecule parameters can be tuned and do not affect each other.