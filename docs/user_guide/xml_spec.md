# How to write XML file

OpenMM的力场文件设计颇为模块化，有着很高的便利性。但遗憾的是现存资料较少，文档不够明确。现将OpenMM XML文件格式及含义整理如下。
拓扑文件
拓扑文件用于描述残基的成键信息。对于残基名称匹配的分子，OpenMM的Topology模块会按照XML文件中信息为原子添加成键。
XML文件示例如下：
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

其中"-C"表示与上一个残基中的"C"原子连接。在匹配时，同名残基内所有原子间均会尝试匹配，一旦匹配成功则会成键，匹配失败则跳过，因此实际成键数可以少于模板中设定的数目。
XML文件注册方法如下：
try:
    import openmm.app as app
except:
    import simtk.openmm.app as app
    
    
app.Topology.loadBondDefinations("residues.xml") # 注册残基拓扑

# 创建Topology并为其添加原子和残基, 读取PDB时自动执行这一过程
top = app.Topology()
...
top.createStandardBonds() # 依照模板文件连接成键

需要注意的是disulfide bond不在这一步骤中完成。OpenMM Topology类会寻找有CYS中没有连接HG的SG原子，并将小于0.3 nm的原子对连接为disulfide bond。
力场参数文件
力场参数文件如下所示：
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

这一文件可以分为残基部分与力场部分。
残基部分
残基部分为：
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

残基部分的<Atom>节点定义了残基中每个原子的atomtype和一些per atom的参数信息，以供力场部分按需调用。其中的<Bond>节点则定义了残基的成键信息，这一部分包含的信息与上文拓扑文件中会有一些区别。以ALA举例，对于ALA，我们通常至少需要定义它的N端、C端和链中三个状态，力场中模板如下：
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

在这一例子中，ALA、CALA、NALA的原子个数与成键关系均不相同。匹配每个ALA时，OpenMM会尝试将CALA、NALA与ALA均进行匹配，最后选择原子个数、元素组成、成键关系均与该残基相同的模板来为每个原子定义力场参数。
力场部分
力场部分包含了以下内容：
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

其中<AtomTypes>节点定义了诸多原子类型。残基部分中每个粒子的"type"标签会与<AtomTypes>各个子节点的"name"标签匹配。对于每个原子类型，它还定义了"class"标签，用于不同的匹配场景。不同<Type>子节点的"name"必须不同，但"class"可以相同。
<XXXForce>节点则定义了某种势函数的匹配规则，如<HarmonicBondForce>定义了Harmonic Bond、<NonbondedForce>节点则定义了分子间相互作用。具体参数细节可以查看文档：
http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html#writing-the-xml-file
匹配过程中，OpenMM会遍历所有的atom、bond、angle、dihedral、improper，并将所有能匹配到的条目全部加入总的势函数中。匹配可以根据"type"标签进行，对应<AtomType>中每个原子的"name"；还可以根据"class"标签进行，对应<AtomType>中每个原子的"class"。这一设计适用于原子类型多但大体相同的情况，譬如小分子力场中LJ参数种类较少，但分子内作用力参数则种类繁多。我们甚至可以对特定的小分子创建单独的type用于定义intra-molecular interaction，但在LJ上则归属于相同的class，以达成小分子参数各自调优、互不影响的效果。