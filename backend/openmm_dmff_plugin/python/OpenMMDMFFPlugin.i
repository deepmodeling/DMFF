%module OpenMMDMFFPlugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>

%inline %{
using namespace std;
%}

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
}

%{
#include "DMFFForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <vector>
%}


/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace DMFFPlugin {

class DMFFForce : public OpenMM::Force {
public:
    DMFFForce(const string& GraphFile);

    void setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient);
    void setHasAux(const bool hasAux);
    void setCutoff(const double cutoff);
    /*
     * Add methods for casting a Force to a DMFFForce.
    */
    %extend {
        static DMFFPlugin::DMFFForce& cast(OpenMM::Force& force) {
            return dynamic_cast<DMFFPlugin::DMFFForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<DMFFPlugin::DMFFForce*>(&force) != NULL);
        }
    }
};

}
