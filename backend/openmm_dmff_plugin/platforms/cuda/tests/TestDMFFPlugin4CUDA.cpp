/* -------------------------------------------------------------------------- *
 *                                   OpenMM-DMFF                              *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "DMFFForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <numeric>


using namespace OpenMM;
using namespace DMFFPlugin;
using namespace std;

extern "C" OPENMM_EXPORT void registerDMFFCudaKernelFactories();

const double TOL = 1e-5;
const string graph = "../python/OpenMMDMFFPlugin/data/lj_fluid_gpu";
const double coordUnitCoeff = 1;
const double forceUnitCoeff = 1;
const double energyUnitCoeff = 1;
const double temperature = 100;
const int randomSeed = 123456;

vector<int64_t> coord_shape = vector<int64_t>(2);
vector<int64_t> box_shape{3, 3};
vector<int64_t> pair_shape = vector<int64_t>(2);
vector<int32_t> pairs_v;
OpenMM::NeighborList neighborList;
vector<std::set<int>> exclusions;

cppflow::tensor coord_tensor, box_tensor, pair_tensor;
vector<cppflow::tensor> output_tensors;
vector<std::string> operations;
vector<std::string> input_node_names = vector<std::string>(3);


void referenceDMFFForce(vector<Vec3> positions, vector<Vec3> box, vector<Vec3>& force, double& energy, cppflow::model dmff_model){
    int natoms = positions.size();
    vector<COORDTYPE> input_coords(natoms*3);
    vector<COORDTYPE> input_box(9);
    vector<FORCETYPE> dmff_force(natoms*3);
    ENERGYTYPE dmff_energy;
    
    // Set box and coordinates input for dmff jax model.
    for (int ii = 0; ii < natoms; ++ii){
        input_coords[ii * 3 + 0] = positions[ii][0] * coordUnitCoeff;
        input_coords[ii * 3 + 1] = positions[ii][1] * coordUnitCoeff;
        input_coords[ii * 3 + 2] = positions[ii][2] * coordUnitCoeff;
    }
    input_box[0] = box[0][0] * coordUnitCoeff;
    input_box[1] = box[0][1] * coordUnitCoeff;
    input_box[2] = box[0][2] * coordUnitCoeff;
    input_box[3] = box[1][0] * coordUnitCoeff;
    input_box[4] = box[1][1] * coordUnitCoeff;
    input_box[5] = box[1][2] * coordUnitCoeff;
    input_box[6] = box[2][0] * coordUnitCoeff;
    input_box[7] = box[2][1] * coordUnitCoeff;
    input_box[8] = box[2][2] * coordUnitCoeff;
    
    // Evaluate and get DMFF forces and energy.
    //nnp_inter.compute (nnp_energy, dmff_force, nnp_virial, input_coords, types, input_box);
    box_tensor = cppflow::tensor(input_box, box_shape);
    coord_tensor = cppflow::tensor(input_coords, coord_shape);


    computeNeighborListVoxelHash(
        neighborList,
        natoms,
        positions,
        exclusions,
        box.data(),
        true,
        1.2,
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

    output_tensors = dmff_model({{input_node_names[0], coord_tensor}, {input_node_names[1], box_tensor}, {input_node_names[2], pair_tensor}}, {"PartitionedCall:0", "PartitionedCall:1"});
    
    dmff_energy = output_tensors[0].get_data<ENERGYTYPE>()[0];
    dmff_force = output_tensors[1].get_data<FORCETYPE>();    


    // Assign the energy and forces as return values.
    energy = static_cast<double>(dmff_energy) * energyUnitCoeff;
    for(int ii = 0; ii < natoms; ++ii){
        force[ii][0] = - dmff_force[ii * 3 + 0] * forceUnitCoeff;
        force[ii][1] = - dmff_force[ii * 3 + 1] * forceUnitCoeff;
        force[ii][2] = - dmff_force[ii * 3 + 2] * forceUnitCoeff;
    }
}

void testDMFFDynamics(int natoms, vector<double> coord, vector<double> box, vector<double> mass, int nsteps=100){
    System system;
    VerletIntegrator integrator(0.0002); // Time step is 0.2 fs here.
    DMFFForce* dmff_force = new DMFFForce(graph);

    // Convert the units of coordinates and box from angstrom to nanometers.
    vector<Vec3> omm_coord;
    vector<Vec3> omm_box;
    for(int ii = 0; ii < 3; ii++){
        omm_box.push_back(Vec3(box[ii * 3 + 0] / coordUnitCoeff, box[ii * 3 + 1] / coordUnitCoeff, box[ii * 3 + 2] / coordUnitCoeff));
    }
    for (int ii = 0; ii < natoms; ++ii){
        system.addParticle(mass[ii]);
        omm_coord.push_back(Vec3(coord[ii * 3 + 0], coord[ii * 3 + 1], coord[ii * 3 + 2]));
    }
    dmff_force->setUnitTransformCoefficients(coordUnitCoeff, forceUnitCoeff, energyUnitCoeff); 
    system.addForce(dmff_force);

    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    context.setPositions(omm_coord);
    context.setPeriodicBoxVectors(omm_box[0], omm_box[1], omm_box[2]);
    context.setVelocitiesToTemperature(temperature, randomSeed);

    // Initialize the jax_model for comparision.
    cppflow::model jax_model = cppflow::model(graph);

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

    coord_shape[0] = natoms;
    coord_shape[1] = 3;
    exclusions.resize(natoms);

    for (int ii = 0; ii < nsteps; ++ii){
        // Running 1 step dynamics.
        integrator.step(1);
        // Get the forces and energy from openmm context state.
        State state = context.getState(State::Forces | State::Energy | State::Positions);
        const vector<Vec3>& omm_forces = state.getForces();
        const double& omm_energy = state.getPotentialEnergy();

        // Calculate the force from jax model directly.
        std::vector<Vec3> forces(natoms, Vec3(0,0,0));
        double energy;
        referenceDMFFForce(state.getPositions(), omm_box, forces, energy, jax_model);

        for (int jj = 0; jj < natoms; ++jj){
            ASSERT_EQUAL_VEC(omm_forces[jj], forces[jj], TOL);
        }
        ASSERT_EQUAL_TOL(energy, omm_energy, TOL);
    }
}


int main(int argc, char* argv[]) {
    // Initialize positions, unit is nanometer.
    std::vector<double> coord = {
        1.4869,1.4417,1.8370000000000002,0.4282,1.2164000000000001,2.3527,0.24500000000000002,1.4031000000000002,0.8695,1.658,0.4821,1.0031,1.1704,1.4292,0.0907,2.3001,1.2364000000000002,0.7593000000000001,2.3508,1.5238,1.5395,0.6515,1.2797,1.1300000000000001,1.3238,1.0387000000000002,0.6028,1.0313,2.3941,1.1909,1.3783,2.2266,1.5460000000000003,2.1957,1.0599,1.455,0.11530000000000001,1.4707000000000001,2.1146000000000003,0.3065,0.5774,1.8188,-0.0494,1.832,2.3497,1.8178999999999998,1.1692,2.0008,1.8654000000000002,0.343,1.8725000000000003,0.4292,1.5221,1.4368,1.3570000000000002,2.0981,1.103,1.4098000000000002,0.2258,0.32630000000000003,0.21230000000000004,0.9242000000000001,2.1351,1.252,0.6546000000000001,1.545,2.4359,1.4137000000000002,0.09910000000000001,0.662,1.3891,0.24460000000000004,2.1807,1.0765,0.3568,1.0917000000000001,0.7743000000000001,2.2414,0.4378,2.1796,0.9539,1.2263000000000002,1.6801,1.136,2.3466,1.5591,1.103,1.2570000000000001,1.1877000000000002,2.164,0.49570000000000003,1.658,0.5198,1.3144,1.4976000000000003,0.7143,1.0516,0.0978,1.6482,2.1533,2.2135000000000002,2.1415,1.9163000000000001,1.5897000000000001,1.2458,1.8677000000000001,0.8567,1.7155000000000002,2.1512000000000002,0.5445000000000001,1.576,1.5814000000000001,1.9201000000000001,1.7932,1.9875,0.7042,2.1085000000000003,1.8557,1.843,2.1122,1.9743,2.0838,1.7328000000000001,1.4769,2.0688,2.3225000000000002,0.15880000000000002,1.8634000000000002,1.31,1.9523000000000001,1.4241000000000001,0.2902,1.7763000000000002,1.2461000000000002,1.5118,2.2309,0.6424000000000001,0.4232,2.0509,0.19720000000000001,2.2418,0.7959,2.2298,1.8864999999999998,0.6643,2.4145000000000003,1.4313000000000002,0.9792000000000001,1.2498,0.5067,1.1904000000000001,1.7758,1.6664000000000003,0.29700000000000004,0.4565000000000001,2.2786000000000004,0.9821,1.8803999999999998,2.061,0.2198,1.2162000000000002,1.7406,0.1378,0.1044,0.9499000000000001,0.20390000000000003,0.5397000000000001,1.0388,1.8989000000000003,1.6082,1.7350000000000003,0.18600000000000003,1.8321000000000003,0.8019000000000001,0.8502000000000001,0.31880000000000003,2.4162,2.0214,0.8935000000000001,0.7367,2.1347,2.326,1.3818000000000001,0.994,0.2096,0.4845,0.2175,2.3638000000000003,1.3552,1.0178,0.08750000000000001,2.1046,0.2683,0.1509,0.2312,0.49800000000000005,1.9023,2.1448,1.2019000000000002,0.6935,1.0732,2.4222,2.1601,1.046,1.5106000000000002,0.9357,1.3374000000000001,1.7486000000000002,0.0001,1.5913000000000002,1.3398,1.6791,2.1634,0.3709,0.9591,0.9917,1.9379000000000002,0.7608,1.2121000000000002,0.25070000000000003,1.2747000000000002,0.42880000000000007,2.3371,0.8711000000000001,1.8224,1.185,1.9267,0.7294,0.7635000000000001,1.5939,2.3087,0.5569000000000001,0.8128000000000001,2.3936,1.5107,0.621,1.8996,0.26110000000000005,1.7603000000000002,1.7589,0.8151,0.8802,0.9716,1.0201,2.1419,0.40990000000000004,1.6098,1.3719000000000001,2.3480000000000003,0.8929,1.5590000000000002,0.8311000000000001,1.3937,0.23870000000000002,1.4025,1.2885,0.2555,0.9979,0.5136999999999999,0.9361000000000002,0.39740000000000003,0.1281,0.862,0.6312000000000001,1.7553999999999998,1.2711000000000001,0.6960000000000001,1.5184,2.2293,0.3469,2.3319,0.4435,1.979,1.0995,0.5888,0.2383,0.0459,0.0884,2.2377,0.7851,2.2165,2.3288,1.6031,0.9092000000000001,0.9029,1.5514000000000001,1.3294000000000001,1.0917000000000001,0.8621000000000001,1.6037,1.361,1.3277,0.5452,0.6697000000000001,0.7398,1.2445000000000002,1.5919,0.12330000000000002,0.9811000000000001,0.1521,1.7182,0.9617000000000001,0.405,2.381,1.586,0.5104000000000001,0.6341000000000001,1.9363000000000001,0.1958,0.48150000000000004,0.9375,1.4548,0.6653,0.5055,0.3047,2.0997,1.8672000000000002,0.21680000000000002,1.9649999999999999,1.1833,1.0909000000000002,1.3763,1.8358,0.33340000000000003,0.6167,0.5750000000000001,1.2102000000000002,2.1995999999999998,1.3109000000000002,1.9009,0.5614,0.1795,1.0621,1.5168,0.6135,0.16970000000000002,0.9818,0.37210000000000004,1.3101000000000003,1.6585999999999999,2.1465,0.77,1.2604,2.2049,1.9687000000000001,1.9338000000000002,0.6234000000000001,0.0946,2.1932,0.5114,0.9361000000000002,0.5063,0.0862,2.1896999999999998,0.49570000000000003,0.20779999999999998,0.6381000000000001,0.23290000000000002,0.5797,0.2647,0.15910000000000002,1.2245,1.2844,1.6597000000000002,2.4419000000000004,1.1129,1.2369,1.3273000000000001,1.4671,0.5469,1.9987,1.3801,1.979,2.2589,0.4699,1.8303000000000003,0.21030000000000004,0.21800000000000003,0.9397000000000001,1.6920000000000002,0.4039,1.4287,0.1847,1.8767,1.5318,1.4136,0.3267,0.3819,2.2102,0.5225,0.9017,0.9943,0.1343,0.0959,0.7195,1.4226,1.8988,1.0612000000000001,0.011000000000000001,0.5231,1.6952000000000003,1.0156,0.15810000000000002,0.13970000000000002,1.7762000000000002,1.3682,1.029,0.17070000000000002,0.5629,0.9455,1.8879000000000001,0.8945000000000001,1.9775,1.088,1.5278,1.645,1.4302000000000001,1.1055,0.4757,1.9054,0.6253000000000001,0.20270000000000002,1.7903,0.7812000000000001,0.6088,1.625,1.6886,0.5251,1.5066000000000002,2.0992,2.4409,0.9244,1.2841,2.3567,0.6889000000000001,0.9853000000000001,0.9608000000000001,1.3817000000000002,0.6080000000000001,1.203,0.6994,1.6666,0.15900000000000003,0.6957,0.5502,1.4368,1.9486999999999999,1.7292000000000003,2.061,1.3492000000000002,1.2589000000000001,0.38680000000000003,2.3253,1.7936,1.8175999999999999,0.5237,0.399,2.1877,1.7484000000000002,1.7109000000000003,1.7693000000000003,0.06530000000000001,0.1459,2.1296,2.0946000000000002,0.3396,2.2007000000000003,0.04000000000000001,0.9349000000000001,0.7859,0.5703,2.2681,1.0914,2.2751,2.2311,1.9684000000000001,0.6532,0.7358,1.9657,0.683,0.8435000000000001,2.3908,0.7913000000000001,2.2823,1.8032000000000001,1.9242000000000001,0.6987000000000001,2.4374000000000002,0.2751,0.12380000000000001,1.9288,0.31520000000000004,0.37660000000000005,0.3412,1.4505000000000001,1.7479,2.3245,2.0271000000000003,0.8131,1.149,2.1734000000000004,1.0833000000000002,1.8968,0.039200000000000006,2.3826,0.2848,1.3407,2.0369,1.5881,0.8055,2.0751000000000004,0.24700000000000003,0.5736,1.129,2.4050000000000002,0.10800000000000001,2.1736999999999997,0.053200000000000004,1.8006000000000002,1.5141,0.0407,2.3854,2.4202000000000004,2.1236,1.6792000000000002,0.9624000000000001,2.0857,1.5029000000000001,1.0552,1.4344000000000001,1.8936000000000002,2.3468,0.6714000000000001,2.3607,1.6348,1.7348,0.9441000000000002,1.9555,0.27740000000000004,1.2697,2.3304,2.2686,0.1339,1.5751,0.8855000000000001,2.2264,1.7631000000000001,0.5546000000000001,0.5404,2.3537,0.8901,1.0565,1.4367,1.0164,1.7736999999999998,1.02,0.9386,1.8329000000000002,0.8833000000000001,0.4201,0.8357,2.4159,2.3893,0.5675,2.1608,1.8838000000000001,1.4112,0.7187000000000001,1.7854,1.7442000000000002,0.34600000000000003,1.6451000000000002,0.042300000000000004,1.8826,2.222,0.052300000000000006,2.2824000000000004,0.2641,0.17490000000000003,1.6754000000000002,0.4199,2.2403,0.0853,0.3877,0.6189,0.7160000000000001,0.5518,2.0741,2.0256000000000003,1.5051,2.2686,0.49340000000000006,0.6245,0.7081000000000001,2.0069,2.0260000000000002,1.0139,1.1265,1.2039,1.013
    };
    std::vector<double> box = {
    2.4413, 0., 0., 0., 2.4413, 0., 0., 0., 2.4413
    };
    
    std::vector<double> mass;
    int nsteps = 100;
    int natoms = coord.size() / 3;
    for(int ii = 0; ii < natoms; ++ii){
        mass.push_back(15.99943);
    }

    // Test the single point energy and dynamics of DMFF Plugin.
    try{
        registerDMFFCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testDMFFDynamics(natoms, coord, box, mass, nsteps);
    }
    catch(const OpenMM::OpenMMException& e) {
        cout << "OpenMMException: "<<e.what() << endl;
        return 1;
    }
    cout<<"Done"<<endl;
    return 0;
}
