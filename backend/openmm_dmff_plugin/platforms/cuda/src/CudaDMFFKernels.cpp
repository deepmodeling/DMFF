/* -------------------------------------------------------------------------- *
 *                                   OpenMM-DMFF                              *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaDMFFKernels.h"
#include "CudaDMFFKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <algorithm>

using namespace DMFFPlugin;
using namespace OpenMM;
using namespace std;


CudaCalcDMFFForceKernel::~CudaCalcDMFFForceKernel(){
   return;
}

void CudaCalcDMFFForceKernel::initialize(const System& system, const DMFFForce& force){
    graph_file = force.getDMFFGraphFile();
    forceUnitCoeff = force.getForceUnitCoefficient();
    energyUnitCoeff = force.getEnergyUnitCoefficient();
    coordUnitCoeff = force.getCoordUnitCoefficient();
    cutoff = force.getCutoff();
    
    natoms = system.getNumParticles();
    coord_shape[0] = natoms;
    coord_shape[1] = 3;
    exclusions.resize(natoms);

    // Load the ordinary graph firstly.
    jax_model.init(graph_file);

    operations = jax_model.get_operations();
    for (int ii = 0; ii < operations.size(); ii++){
        if (operations[ii].find("serving")!= std::string::npos){
            if (operations[ii].find("0")!= std::string::npos){
                input_node_names[0] = operations[ii] + ":0";
            } else if (operations[ii].find("1") != std::string::npos){
                input_node_names[1] = operations[ii] + ":0";
            } else if (operations[ii].find("2") != std::string::npos){
                input_node_names[2] = operations[ii] + ":0";
            }
        }
    }

    // Initialize the ordinary input and output array.
    // Initialize the input tensor.
    dener = 0.;
    dforce = vector<FORCETYPE>(natoms * 3, 0.);
    dcoord = vector<COORDTYPE>(natoms * 3, 0.);
    dbox = vector<COORDTYPE>(9, 0.);
        
    AddedForces = vector<double>(natoms * 3, 0.0);
    // Set for CUDA context.
    cu.setAsCurrent();
    map<string, string> defines;
    defines["FORCES_TYPE"] = "double";
    dmffForces.initialize(cu, 3*natoms, sizeof(double), "dmffForces");
    CUmodule module = cu.createModule(CudaDMFFKernelSources::DMFFForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}


double CudaCalcDMFFForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    Vec3 box[3];

    // Set box size.
    if ( !context.getSystem().usesPeriodicBoundaryConditions() ){
        dbox = {}; // No PBC.
        throw OpenMMException("DMFFForce requires periodic boundary conditions.");
    }
    
    cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
    // Transform unit from nanometers to the required units in DMFF input.
    dbox[0] = box[0][0] * coordUnitCoeff;
    dbox[1] = box[0][1] * coordUnitCoeff;
    dbox[2] = box[0][2] * coordUnitCoeff;
    dbox[3] = box[1][0] * coordUnitCoeff;
    dbox[4] = box[1][1] * coordUnitCoeff;
    dbox[5] = box[1][2] * coordUnitCoeff;
    dbox[6] = box[2][0] * coordUnitCoeff;
    dbox[7] = box[2][1] * coordUnitCoeff;
    dbox[8] = box[2][2] * coordUnitCoeff;
    cppflow::tensor box_tensor = cppflow::tensor(dbox, box_shape);

    // Set input coord.
    for(int ii = 0; ii < natoms; ++ii){
        // Multiply by coordUnitCoeff to transform unit from nanometers to input units for DMFF model.
        dcoord[ii * 3 + 0] = pos[ii][0] * coordUnitCoeff;
        dcoord[ii * 3 + 1] = pos[ii][1] * coordUnitCoeff;
        dcoord[ii * 3 + 2] = pos[ii][2] * coordUnitCoeff;
    }
    coord_tensor = cppflow::tensor(dcoord, coord_shape);

    // Fetch the neighbor list for input pairs tensor.
    computeNeighborListVoxelHash(
        neighborList,
        natoms,
        pos,
        exclusions,
        box,
        true,
        cutoff,
        0.0
    );
    int totpairs = neighborList.size();
    pairs_v = vector<int32_t>(totpairs * 2);
    for (int ii = 0; ii < totpairs; ii ++){
        pairs_v[ ii * 2 + 0 ] = neighborList[ii].second;
        pairs_v[ ii * 2 + 1 ] = neighborList[ii].first;
    }
    pair_shape[0] = totpairs;
    pair_shape[1] = 2;
    pair_tensor = cppflow::tensor(pairs_v, pair_shape);

    // Calculate the energy and forces.
    output_tensors = jax_model({{input_node_names[0], coord_tensor}, {input_node_names[1], box_tensor}, {input_node_names[2], pair_tensor}}, {"PartitionedCall:0", "PartitionedCall:1"});
    
    dener = output_tensors[0].get_data<ENERGYTYPE>()[0];
    dforce = output_tensors[1].get_data<FORCETYPE>();    
    
    
    // Transform the unit from eV/A to KJ/(mol*nm)
    for(int ii = 0; ii < natoms; ii ++){
        AddedForces[ii * 3 + 0] = - dforce[ii * 3 + 0] * forceUnitCoeff;
        AddedForces[ii * 3 + 1] = - dforce[ii * 3 + 1] * forceUnitCoeff;
        AddedForces[ii * 3 + 2] = - dforce[ii * 3 + 2] * forceUnitCoeff;
    }
    // Transform the unit from eV to KJ/mol
    dener = dener * energyUnitCoeff;

    if (includeForces) {
        // Change to OpenMM CUDA context.
        cu.setAsCurrent();
        dmffForces.upload(AddedForces);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&dmffForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &natoms, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, natoms);
    }
    if (!includeEnergy){
        dener = 0.0;
    }
    return dener;
}



