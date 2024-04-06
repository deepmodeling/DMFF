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

#include "DMFFForce.h"
#include "internal/DMFFForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <sys/stat.h>

using namespace DMFFPlugin;
using namespace OpenMM;
using namespace std;


DMFFForce::DMFFForce(const string& GraphFile){
    graph_file  = GraphFile;
}

DMFFForce::~DMFFForce(){
    return;
}

void DMFFForce::setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient){
    coordCoeff = coordCoefficient;
    forceCoeff = forceCoefficient;
    energyCoeff = energyCoefficient;
}

void DMFFForce::setHasAux(const bool hasAux){
    this->has_aux = hasAux;
}

void DMFFForce::setCutoff(const double cutoff){
    this->cutoff = cutoff;
}

double DMFFForce::getCoordUnitCoefficient() const {return coordCoeff;}
double DMFFForce::getForceUnitCoefficient() const {return forceCoeff;}
double DMFFForce::getEnergyUnitCoefficient() const {return energyCoeff;}

double DMFFForce::getCutoff() const {return cutoff;}

bool DMFFForce::getHasAux() const {return has_aux;}

const string& DMFFForce::getDMFFGraphFile() const{return graph_file;}



ForceImpl* DMFFForce::createImpl() const {
    return new DMFFForceImpl(*this);
}

void DMFFForce::updateParametersInContext(Context& context) {
    // Nothing to be done here.
    return;
}

