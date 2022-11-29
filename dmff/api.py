import linecache
from typing import Callable, Dict, Any

import numpy as np
import jax.numpy as jnp

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from dmff.utils import DMFFException
from dmff.fftree import ForcefieldTree, XMLParser


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def build_covalent_map(data, max_neighbor):
    n_atoms = len(data.atoms)
    covalent_map = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in data.bonds:
        covalent_map[bond.atom1, bond.atom2] = 1
        covalent_map[bond.atom2, bond.atom1] = 1
    for n_curr in range(1, max_neighbor):
        for i in range(n_atoms):
            # current neighbors
            j_list = np.where(
                np.logical_and(covalent_map[i] <= n_curr,
                               covalent_map[i] > 0))[0]
            for j in j_list:
                k_list = np.where(covalent_map[j] == 1)[0]
                for k in k_list:
                    if k != i and k not in j_list:
                        covalent_map[i, k] = n_curr + 1
                        covalent_map[k, i] = n_curr + 1
    return jnp.array(covalent_map)


def findAtomTypeTexts(attribs, num):
    typetxt = []
    for n in range(1, num + 1):
        for key in ["type%i" % n, "class%i" % n]:
            if key in attribs:
                typetxt.append((key, attribs[key]))
                break
    return typetxt


jaxGenerators = {}


class Potential:
    def __init__(self):
        self.dmff_potentials = {}
        self.omm_system = None

    def addDmffPotential(self, name, potential):
        self.dmff_potentials[name] = potential

    def addOmmSystem(self, system):
        self.omm_system = system

    def buildOmmContext(self, integrator=mm.VerletIntegrator(0.1)):
        if self.omm_system is None:
            raise DMFFException(
                "OpenMM system is not initialized in this object.")
        self.omm_context = mm.Context(self.omm_system, integrator)

    def getPotentialFunc(self, names=[]):
        if len(self.dmff_potentials) == 0:
            raise DMFFException("No DMFF function in this potential object.")

        def totalPE(positions, box, pairs, params):
            totale_list = [
                self.dmff_potentials[k](positions, box, pairs, params)
                for k in self.dmff_potentials.keys()
                if (len(names) == 0 or k in names)
            ]
            totale = jnp.sum(jnp.array(totale_list))
            return totale

        return totalPE


class Hamiltonian(app.forcefield.ForceField):
    def __init__(self, *xmlnames, **kwargs):
        super().__init__(*xmlnames)
        self._pseudo_ff = app.ForceField(*xmlnames)
        # parse XML forcefields
        self.fftree = ForcefieldTree('ForcefieldTree')
        self.xmlparser = XMLParser(self.fftree)
        self.xmlparser.parse(*xmlnames)

        self._jaxGenerators = []
        self._potentials = []
        self.paramtree = {}

        self.ommsys = None

        for child in self.fftree.children:
            if child.tag in jaxGenerators:
                self._jaxGenerators.append(jaxGenerators[child.tag](self))

        # initialize paramtree
        self.extractParameterTree()

        # hook generators to self._forces
        # use noOmmSys to disable all traditional openmm system
        if kwargs.get("noOmmSys", False):
            self._forces = []
        for jaxGen in self._jaxGenerators:
            self._forces.append(jaxGen)

    def getGenerators(self):
        return self._jaxGenerators

    def extractParameterTree(self):
        # load Force info
        for jaxgen in self._jaxGenerators:
            jaxgen.extract()

    def overwriteParameterTree(self):
        # write Force info
        for jaxgen in self._jaxGenerators:
            jaxgen.overwrite()
        pass

    def createPotential(self,
                        topology,
                        nonbondedMethod=app.NoCutoff,
                        nonbondedCutoff=1.0 * unit.nanometer,
                        jaxForces=[],
                        **args):
        # load_constraints_from_system_if_needed
        # create potentials
        """
        Create differentiable jax potential for given openmm.app.Topology object

        Parameters
        ----------
        topology: openmm.app.Topology
            Input openmm topology
        nonbondedMethod: object=NoCutoff
            The method to use for nonbonded interactions. Allowed values are 
            NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME.
        nonbondedCutoff : distance=1*nanometer
            The cutoff distance to use for nonbonded interactions
        jaxForces: list of str
            Specified forces to create. If set to [], will create all existing types of forces.
        args
            Arbitrary parameters in openmm.app.ForceField.createSystem function
        
        Return
        ------
        potObj: dmff.api.Potential
            Differentiable jax potential energy function
        """
        pseudo_data = app.ForceField._SystemData(topology)
        residueTemplates = {}
        templateForResidue = self._pseudo_ff._matchAllResiduesToTemplates(pseudo_data, topology, residueTemplates, False)
        self.templateNameForResidue = [i.name for i in templateForResidue]

        system = self.createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            **args,
        )
        removeIdx = []
        jaxGens = [i.name for i in self._jaxGenerators]
        for nf, force in enumerate(system.getForces()):
            if (len(jaxForces) > 0
                    and force.getName() in jaxForces) or (force.getName()
                                                          in jaxGens):
                removeIdx.append(nf)
        for nf in removeIdx[::-1]:
            system.removeForce(nf)

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

            # virtual site
            try:
                addVsiteFunc = generator.getAddVsiteFunc()
                self.setAddVirtualSiteFunc(addVsiteFunc)
                vsiteObj = generator.getVsiteObj()
                self.setVirtualSiteObj(vsiteObj)
            except AttributeError as e:
                pass

            # covalent map
            try:
                cov_map = generator.covalent_map
                self.setCovalentMap(cov_map)
            except AttributeError as e:
                pass

            # topology matrix (for BCC usage)
            try:
                top_mat = generator.getTopologyMatrix()
                self.setTopologyMatrix(top_mat)
            except AttributeError as e:
                pass

        return potObj

    def render(self, filename):
        self.overwriteParameterTree()
        self.xmlparser.write(filename)

    def getParameters(self):
        return self.paramtree

    def updateParameters(self, paramtree):
        def update_iter(node, ref):
            for key in ref:
                if isinstance(ref[key], dict):
                    update_iter(node[key], ref[key])
                else:
                    node[key] = ref[key]

        update_iter(self.paramtree, paramtree)
        
    def setCovalentMap(self, cov_map: jnp.ndarray):
        self._cov_map = cov_map
    
    def getCovalentMap(self) -> jnp.ndarray:
        """
        Get covalent map
        """
        if hasattr(self, "_cov_map"):
            return self._cov_map
        else:
            raise DMFFException("Covalent map is not set.")
    
    def getAddVirtualSiteFunc(self) -> Callable:
        return self._add_vsite_coords
    
    def setAddVirtualSiteFunc(self, func: Callable):
        self._add_vsite_coords = func
    
    def setVirtualSiteObj(self, vsite):
        self._vsite = vsite
    
    def getVirtualSiteObj(self):
        return self._vsite
    
    def setTopologyMatrix(self, top_mat):
        self._top_mat = top_mat
    
    def getTopologyMatrix(self):
        return self._top_mat
    
    def addVirtualSiteCoords(self, pos: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Add coordinates for virtual sites

        Parameters
        ----------
        pos: jnp.ndarray
            Coordinates without virtual sites
        params: dict
            Paramtree of hamiltonian, i.e. `dmff.Hamiltonian.paramtree`
        
        Return
        ------
        newpos: jnp.ndarray

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import openmm.app as app
        >>> from rdkit import Chem
        >>> from dmff import Hamiltonian
        >>> pdb = app.PDBFile("tests/data/chlorobenzene.pdb")
        >>> pos = jnp.array(pdb.getPositions(asNumpy=True)._value)
        >>> mol = Chem.MolFromMolFile("tests/data/chlorobenzene.mol", removeHs=False)
        >>> h = Hamiltonian("tests/data/cholorobenzene_vsite.xml")
        >>> potObj = h.createPotential(pdb.topology, rdmol=mol)
        >>> newpos = h.addVirtualSiteCoords(pos, h.paramtree)
        
        """
        func = self.getAddVirtualSiteFunc()
        newpos = func(pos, params)
        return newpos
    
    def addVirtualSiteToMol(self, rdmol, params):
        """
        Add coordinates for rdkit.Chem.Mol object

        Parameters
        ----------
        rdmol: rdkit.Chem.Mol
            Mol object to which virtual sites are added
        params: dict
            Paramtree of hamiltonian, i.e. `dmff.Hamiltonian.paramtree`
        
        Return
        ------
        newmol: rdkit.Chem.Mol
            Mol object with virtual sites added
        
        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import openmm.app as app
        >>> from rdkit import Chem
        >>> from dmff import Hamiltonian
        >>> pdb = app.PDBFile("tests/data/chlorobenzene.pdb")
        >>> mol = Chem.MolFromMolFile("tests/data/chlorobenzene.mol", removeHs=False)
        >>> h = Hamiltonian("tests/data/cholorobenzene_vsite.xml")
        >>> potObj = h.createPotential(pdb.topology, rdmol=mol)
        >>> newmol = h.addVirtualSiteToMol(mol, h.paramtree)
        """
        vsiteObj = self.getVirtualSiteObj()
        newmol = vsiteObj.addVirtualSiteToMol(
            rdmol,
            params['NonbondedForce']['vsite_types'],
            params['NonbondedForce']['vsite_distances']
        )
        return newmol
    
    @staticmethod
    def buildTopologyFromMol(rdmol, resname: str = "MOL") -> app.Topology:
        """
        Build openmm.app.Topology from rdkit.Chem.Mol Object

        Parameters
        ----------
        rdmol: rdkit.Chem.Mol
            Mol object
        resname: str
            Name of the added residue, default "MOL"
        
        Return
        ------
        top: `openmm.app.Topology`
            Topology built based on the input rdkit Mol object
        """
        from rdkit import Chem

        top = app.Topology()
        chain = top.addChain(0)
        res = top.addResidue(resname, chain, "1", "")
        
        atCount = {}
        addedAtoms = []
        for idx, atom in enumerate(rdmol.GetAtoms()):
            symb = atom.GetSymbol().upper()
            atCount.update({symb: atCount.get(symb, 0) + 1})
            ele = app.Element.getBySymbol(symb)
            atName = f'{symb}{atCount[symb]}'
        
            addedAtom = top.addAtom(atName, ele, res, str(idx+1))
            addedAtoms.append(addedAtom)
        
        bondTypeMap = {
            Chem.rdchem.BondType.SINGLE: app.Single,
            Chem.rdchem.BondType.DOUBLE: app.Double,
            Chem.rdchem.BondType.TRIPLE: app.Triple,
            Chem.rdchem.BondType.AROMATIC: app.Aromatic
        }
        for bond in rdmol.GetBonds():
            top.addBond(
                addedAtoms[bond.GetBeginAtomIdx()],
                addedAtoms[bond.GetEndAtomIdx()],
                type=bondTypeMap.get(bond.GetBondType(), None)
            )
        return top