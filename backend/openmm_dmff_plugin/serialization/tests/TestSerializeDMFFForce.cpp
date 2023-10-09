/* -------------------------------------------------------------------------- *
 *                                OpenMM-DMFF                                 *
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

#include "DMFFForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace DMFFPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerDMFFSerializationProxies();


void testSerialization() {
    const double TOL = 1e-5;
    const string graph = "../python/OpenMMDMFFPlugin/data/lj_fluid_gpu";
    const double temperature = 300;

    // Create a Force.
    DMFFForce dmff_force = DMFFForce(graph);
    
    stringstream buffer;
    XmlSerializer::serialize<DMFFForce>(&dmff_force, "Force", buffer);
    DMFFForce* copy = XmlSerializer::deserialize<DMFFForce>(buffer);

    // Compare the two forces to see if they are identical.
    DMFFForce& dmff_force2 = *copy;
    ASSERT_EQUAL(dmff_force2.getDMFFGraphFile(), dmff_force.getDMFFGraphFile());
    return;
}

int main() {
    try {
        registerDMFFSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
