import sys
import json
from rdkit import Chem

inp = sys.argv[1]
out = sys.argv[2]

m = Chem.MolFromMolFile(inp, sanitize=False, removeHs=False, strictParsing=False)
bond_patt = Chem.MolFromSmarts("[*:1]~[*:2]")
angle_patt = Chem.MolFromSmarts("[*:1]~[*:2]~[*:3]")
dihedral_patt = Chem.MolFromSmarts("[*:1]~[*:2]~[*:3]~[*:4]")
improper_patt = Chem.MolFromSmarts("[*:2]~[*:1](~[*:3])~[*:4]")

nbonds = len(m.GetSubstructMatches(bond_patt))
nangles = len(m.GetSubstructMatches(angle_patt))
nprops = len(m.GetSubstructMatches(dihedral_patt))
nimprs = len(m.GetSubstructMatches(improper_patt))
data = {
    "nbonds": nbonds, "nangles": nangles, "nprops": nprops, "nimprs": nimprs
}
with open(out, "w") as f:
    f.write(json.dumps(data))