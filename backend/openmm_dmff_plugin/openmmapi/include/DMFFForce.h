#ifndef OPENMM_DMFFFORCE_H_
#define OPENMM_DMFFFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
// Include cppflow header files for model load and evaluation.
#include <cppflow/ops.h>
#include <cppflow/model.h>

#include "internal/windowsExportDMFF.h"

using namespace std;

#if HIGH_PRECISION
typedef double FORCETYPE;
typedef double COORDTYPE;
typedef double ENERGYTYPE;
#else
typedef float FORCETYPE;
typedef float COORDTYPE;
typedef double ENERGYTYPE;
#endif

namespace DMFFPlugin {
class OPENMM_EXPORT_DMFF DMFFForce : public OpenMM::Force {
public:
    /**
     * @brief Construct a new DMFF Force object. Used for NVT/NPT/NVE simulations.
     * 
     * @param GraphFile 
     */
    DMFFForce(const string& GraphFile);
    ~DMFFForce();
    /**
     * @brief Set the Unit Transform Coefficients.
     * 
     * @param coordCoefficient :  the coordinate transform coefficient.
     * @param forceCoefficient : the force transform coefficient.
     * @param energyCoefficient : the energy transform coefficient.
     */
    void setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient);
    /**
     * @brief Set the has_aux flag when model was saved with auxilary input.
     * 
     * @param hasAux  : true if model was saved with auxilary input.
     */
    void setHasAux(const bool hasAux);
    /**
     * @brief Set the Cutoff for neighbor list fetching.
     * 
     * @param cutoff 
     */
    void setCutoff(const double cutoff);
    /**
     * @brief get the DMFF graph file.
     * 
     * @return const std::string& 
     */
    const std::string& getDMFFGraphFile() const;
    /**
     * @brief Get the Coord Unit Coefficient.
     * 
     * @return double 
     */
    double getCoordUnitCoefficient() const;
    /**
     * @brief Get the Force Unit Coefficient.
     * 
     * @return double 
     */
    double getForceUnitCoefficient() const;
    /**
     * @brief Get the Energy Unit Coefficient.
     * 
     * @return double 
     */
    double getEnergyUnitCoefficient() const;
    /**
     * @brief Get the Cutoff radius of the model used.
     * 
     * @return double
     */
    double getCutoff() const;
    /**
     * @brief Get the Has Aux object
     * 
     * @return true 
     * @return false 
     */
    bool getHasAux() const;
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        return use_pbc;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    string graph_file;
    bool use_pbc = true;
    bool has_aux = false;
    double cutoff = 1.2;
    double coordCoeff, forceCoeff, energyCoeff;
    
};

} // namespace DMFFPlugin

#endif /*OPENMM_DMFFFORCE_H_*/
