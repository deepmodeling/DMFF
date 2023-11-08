/* -------------------------------------------------------------------------- *
 *                                   OpenMM-DMFF                              *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "ReferenceDMFFKernels.h"
#include "DMFFForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include <typeinfo>
#include <iostream>
#include <map>
#include <algorithm>
#include <limits>

using namespace DMFFPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcDMFFForceKernel::~ReferenceCalcDMFFForceKernel(){
    return;
}

void ReferenceCalcDMFFForceKernel::initialize(const System& system, const DMFFForce& force) {
    graph_file = force.getDMFFGraphFile();
    forceUnitCoeff = force.getForceUnitCoefficient();
    energyUnitCoeff = force.getEnergyUnitCoefficient();
    coordUnitCoeff = force.getCoordUnitCoefficient();
    cutoff = force.getCutoff();
    this->has_aux = force.getHasAux();

    natoms = system.getNumParticles();
    coord_shape[0] = natoms;
    coord_shape[1] = 3;
    exclusions.resize(natoms);

    if (this->has_aux){
        U_ind_shape[0] = natoms;
        U_ind_shape[1] = 3;
        // Initialize the last_U_ind.
        for(int ii = 0; ii < natoms * 3; ii ++){
            last_U_ind.push_back(0.0);
        }
    }

    // Load the ordinary graph firstly.
    jax_model.init(graph_file);

    operations = jax_model.get_operations();
    for (int ii = 0; ii < operations.size(); ii++){
        if (operations[ii].find("serving")!= std::string::npos){
            if (operations[ii].find("0")!= std::string::npos){
                input_node_names[0] = operations[ii]+":0";
            } else if (operations[ii].find("1") != std::string::npos){
                input_node_names[1] = operations[ii]+":0";
            } else if (operations[ii].find("2") != std::string::npos){
                input_node_names[2] = operations[ii]+":0";
            } 
            // Set up the auxilary input node name. For U_ind
            if(this->has_aux){
                if (operations[ii].find("3") != std::string::npos){
                    input_node_names.push_back(operations[ii] + ":0");
                }
            }
        }
        // Set up the output names.
        if (operations[ii].find("PartitionedCall") != std::string::npos){
            output_node_names[0] = operations[ii] + ":0";
            output_node_names[1] = operations[ii] + ":1";
            if(this->has_aux){
                output_node_names.push_back(operations[ii] + ":2");
            }
            break;
        }
    }

    // Initialize the ordinary input and output array.
    // Initialize the input tensor.
    dener = 0.;
    dforce = vector<FORCETYPE>(natoms * 3, 0.);
    dcoord = vector<COORDTYPE>(natoms * 3, 0.);
    dbox = vector<COORDTYPE>(9, 0.);
    
    AddedForces = vector<double>(natoms * 3, 0.0);
}

double ReferenceCalcDMFFForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    // Extract the box size.
    if ( ! context.getSystem().usesPeriodicBoundaryConditions()){
        dbox = {}; // No PBC.
        throw OpenMMException("No PBC is not supported yet.");
    }
    Vec3* box = extractBoxVectors(context);
    // Transform unit from nanometers to required units for DMFF model input.
    dbox[0] = box[0][0] * coordUnitCoeff;
    dbox[1] = box[0][1] * coordUnitCoeff;
    dbox[2] = box[0][2] * coordUnitCoeff;
    dbox[3] = box[1][0] * coordUnitCoeff;
    dbox[4] = box[1][1] * coordUnitCoeff;
    dbox[5] = box[1][2] * coordUnitCoeff;
    dbox[6] = box[2][0] * coordUnitCoeff;
    dbox[7] = box[2][1] * coordUnitCoeff;
    dbox[8] = box[2][2] * coordUnitCoeff;
    box_tensor = cppflow::tensor(dbox, box_shape);
    
    // Set input coord.
    for(int ii = 0; ii < natoms; ++ii){
        // Multiply by coordUnitCoeff means the transformation of the unit from nanometers to input units.
        dcoord[ii * 3 + 0] = pos[ii][0] * coordUnitCoeff;
        dcoord[ii * 3 + 1] = pos[ii][1] * coordUnitCoeff;
        dcoord[ii * 3 + 2] = pos[ii][2] * coordUnitCoeff;
    }
    coord_tensor = cppflow::tensor(dcoord, coord_shape);

    // Set input U_ind.
    U_ind_tensor = cppflow::tensor(last_U_ind, U_ind_shape);

    // Set input pairs.
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
    dpairs = vector<int32_t>(totpairs * 2);
    for (int ii = 0; ii < totpairs; ii++)
    {
        int32_t i1 = neighborList[ii].second;
        int32_t i2 = neighborList[ii].first;
        dpairs[ii * 2 + 0 ] = i1;
        dpairs[ii * 2 + 1 ] = i2;
    }
    pair_shape[0] = totpairs;
    pair_shape[1] = 2;
    pair_tensor = cppflow::tensor(dpairs, pair_shape);

    if (!this->has_aux){
        output_tensors = jax_model({
            {input_node_names[0], coord_tensor}, 
            {input_node_names[1], box_tensor}, 
            {input_node_names[2], pair_tensor}}, 
            {output_node_names[0], output_node_names[1]});
        dener = output_tensors[0].get_data<ENERGYTYPE>()[0];
        dforce = output_tensors[1].get_data<FORCETYPE>();    
    } else {
        output_tensors = jax_model({
            {input_node_names[0], coord_tensor}, 
            {input_node_names[1], box_tensor}, 
            {input_node_names[2], U_ind_tensor}, 
            {input_node_names[3], pair_tensor}}, 
            {output_node_names[0], output_node_names[1], output_node_names[2]});

        dener = output_tensors[0].get_data<ENERGYTYPE>()[0];
        dforce = output_tensors[1].get_data<FORCETYPE>();
        // Save last U_ind for next step usage.
        last_U_ind = output_tensors[2].get_data<double>();
    }


    // Transform the unit from output units to KJ/(mol*nm)
    for(int ii = 0; ii < natoms; ii ++){
        AddedForces[ii * 3 + 0] = - dforce[ii * 3 + 0] * forceUnitCoeff;
        AddedForces[ii * 3 + 1] = - dforce[ii * 3 + 1] * forceUnitCoeff;
        AddedForces[ii * 3 + 2] = - dforce[ii * 3 + 2] * forceUnitCoeff;
    }
    // Transform the unit from output units to KJ/mol
    dener = dener * energyUnitCoeff;

    if(includeForces){
        for(int ii = 0; ii < natoms; ii ++){
        force[ii][0] += AddedForces[ii * 3 + 0];
        force[ii][1] += AddedForces[ii * 3 + 1];
        force[ii][2] += AddedForces[ii * 3 + 2];
        }
    }
    if (!includeEnergy){
        dener = 0.0;
    }
    // Return energy.
    return dener;
}


