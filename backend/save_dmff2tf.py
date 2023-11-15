import dmff
from dmff import NeighborList
import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
# The model is saved in float32 precision by default. 
#from jax import config
#config.update("jax_enable_x64", True)
import openmm.app as app
import openmm.unit as unit
import tensorflow as tf

import os
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def create_dmff_potential(input_pdb_file, ff_xml_files, bond_definitions_xml = None, has_aux = False):
    pdb = app.PDBFile(input_pdb_file)
    h = dmff.Hamiltonian(*ff_xml_files)
    if bond_definitions_xml is not None:
        app.Topology.loadBondDefinitions(bond_definitions_xml)
    
    if has_aux:# Used when using ADMP with DMFF.
        pot = h.createPotential(pdb.topology,
                            nonbondedMethod=app.PME,
                            ethresh=5e-4, step_pol=10,
                            nonbondedCutoff=1.2 *
                            unit.nanometer,
                            has_aux=True)
    else:
        pot = h.createPotential(pdb.topology,
                            nonbondedMethod=app.PME,
                            nonbondedCutoff=1.2 *
                            unit.nanometer)
        
    pot_func = pot.getPotentialFunc()
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    a = a.value_in_unit(unit.nanometer)
    b = b.value_in_unit(unit.nanometer)
    c = c.value_in_unit(unit.nanometer)

    if has_aux:
        engrad = jax.value_and_grad(pot_func, 0, has_aux=True)
    else:
        engrad = jax.value_and_grad(pot_func, 0)
    
    covalent_map = pot.meta["cov_map"]
    aux = dict()

    if has_aux:
        def potential_engrad(positions, box, U_ind, pairs):
            if jnp.shape(pairs)[-1] == 2:
                nbond = covalent_map[pairs[:, 0], pairs[:, 1]]
                pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
            aux['U_ind'] = U_ind
            ener_and_aux, ener_grad = engrad(positions, box, pairs, h.getParameters(), aux)
            # Return energy, gradient (forces), and U_ind
            return ener_and_aux[0], ener_grad, ener_and_aux[1]['U_ind']    
    else:
        def potential_engrad(positions, box, pairs):
            if jnp.shape(pairs)[-1] == 2:
                nbond = covalent_map[pairs[:, 0], pairs[:, 1]]
                pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
            
            return engrad(positions, box, pairs, h.getParameters())

    return pdb, potential_engrad, covalent_map, pot, h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdb", dest="input_pdb", help="input pdb file. Box information is required in the pdb file.")
    parser.add_argument("--xml_files", dest="xml_files", nargs="+", help=".xml files with parameters are derived from DMFF.")
    parser.add_argument("--output", dest="output", help="output directory")
    parser.add_argument("--bond_definitions_xml", dest="bond_definitions_xml", help=".xml file that contains bond definitions. Optional", default=None)
    parser.add_argument("--has_aux", dest="has_aux", default=False, help="Enable aux output in the model. Used when using ADMP with DMFF, and the output would be U_ind.")
    args = parser.parse_args()

    input_pdb = args.input_pdb
    ff_xml_files = args.xml_files
    output_dir = args.output
    has_aux = args.has_aux
    bond_definitions_xml = args.bond_definitions_xml
    
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    pdb, pot_grad, covalent_map, pot, h = create_dmff_potential(input_pdb, ff_xml_files, bond_definitions_xml=bond_definitions_xml, has_aux=has_aux)

    natoms = pdb.getTopology().getNumAtoms()

    if has_aux:
        f_tf = jax2tf.convert(
            jax.jit(pot_grad),
            polymorphic_shapes=["("+str(natoms)+", 3)", "(3, 3)", "("+str(natoms)+", 3)", "(b, 2)"]
        )
    else:
        f_tf = jax2tf.convert(
            jax.jit(pot_grad),
            polymorphic_shapes=["("+str(natoms)+", 3)", "(3, 3)", "(b, 2)"]
        )

    dmff_model = tf.Module()
    if has_aux:
        dmff_model.f = tf.function(f_tf, autograph=False,
                            input_signature=[
                                tf.TensorSpec(shape=[natoms,3], dtype=tf.float32), 
                                tf.TensorSpec(shape=[3,3], dtype=tf.float32), 
                                tf.TensorSpec(shape=[natoms,3], dtype=tf.float64), 
                                tf.TensorSpec(shape=tf.TensorShape([None, 2]), dtype=tf.int32)])
    else:
        dmff_model.f = tf.function(f_tf, autograph=False,
                            input_signature=[
                                tf.TensorSpec(shape=[natoms,3], dtype=tf.float32), tf.TensorSpec(shape=[3,3], dtype=tf.float32), 
                                tf.TensorSpec(shape=tf.TensorShape([None, 2]), dtype=tf.int32)
                                ])
    
    tf.saved_model.save(dmff_model, output_dir, options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
