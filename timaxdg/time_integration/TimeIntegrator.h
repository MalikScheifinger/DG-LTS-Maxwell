#ifndef TIME_INTEGRATION_TIMEINTEGRATOR_H_
#define TIME_INTEGRATION_TIMEINTEGRATOR_H_

using namespace dealii;

#include <memory>

namespace MaxwellProblem::TimeIntegration {

/**
* \brief Base class for time integration schemes.
*
* This class is only an abstract class (interface) for all time integration classes.
*/


template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype=Vector<double>>
class TimeIntegrator {
 public:

  /**
   *
   * This routine computes one complete time step with the time integration scheme.
   */
  virtual void integrate_step(
      Vectortype &,
      Vectortype &,
      Vectortype &) = 0;

};

} // MaxwellProblem::TimeIntegration

#endif // TIME_INTEGRATION_TIMEINTEGRATOR_H_
