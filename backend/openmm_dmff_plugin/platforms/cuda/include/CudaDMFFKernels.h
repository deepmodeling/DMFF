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
#ifndef CUDA_DMFF_KERNELS_H_
#define CUDA_DMFF_KERNELS_H_

#include "DMFFKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/reference/ReferenceNeighborList.h"


namespace DMFFPlugin {

/**
 * This kernel is invoked by DMFFForceImpl to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcDMFFForceKernel : public CalcDMFFForceKernel{
public:
    CudaCalcDMFFForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu):CalcDMFFForceKernel(name, platform), cu(cu){};
    ~CudaCalcDMFFForceKernel();
    void initialize(const OpenMM::System& system, const DMFFForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    // Used for CUDA Platform.
    bool hasInitialized;
    OpenMM::CudaContext& cu;
    OpenMM::CudaArray dmffForces;
    CUfunction addForcesKernel;

    std::string graph_file;
    cppflow::model jax_model;
    vector<int64_t> coord_shape = vector<int64_t>(2);
    vector<int64_t> U_ind_shape = vector<int64_t>(2);
    vector<int64_t> box_shape{3, 3};
    vector<int64_t> pair_shape = vector<int64_t>(2);
    vector<int32_t> pairs_v;
    cppflow::tensor coord_tensor, box_tensor, pair_tensor, U_ind_tensor;
    vector<cppflow::tensor> output_tensors;
    vector<double> last_U_ind;
    vector<std::string> operations;
    vector<std::string> input_node_names = vector<std::string>(3);
    vector<std::string> output_node_names = vector<std::string>(2);

    OpenMM::NeighborList neighborList;
    vector<std::set<int>> exclusions;
    
    int natoms;
    double cutoff;
    bool has_aux;
    ENERGYTYPE dener;
    vector<FORCETYPE> dforce;
    vector<COORDTYPE> dcoord;
    vector<COORDTYPE> dbox;
    double forceUnitCoeff, energyUnitCoeff, coordUnitCoeff;
    vector<double> AddedForces;    

};


} // namespace DMFFPlugin


#endif /*CUDA_DMFF_KERNELS_H_*/