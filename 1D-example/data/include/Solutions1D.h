#ifndef SOLUTIONS_1D_H_
#define SOLUTIONS_1D_H_

#include <deal.II/base/function.h>

namespace MaxwellProblem1D {
namespace Data {

/**
 * @brief Cavity solution for 1D mode.
 * 
 * This class implements en exact solution for maxwell equations
 * in 1D mode.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class CavitySolution1D : public dealii::Function<1, double> {
 public:
  /**
   * @brief Construct a new Cavity Solution 1D object
   *  
   * @param mu Permeability
   * @param eps Permittivity
   * @param current_time Time function gets initialized with
   * @param ax X-dimension parameter [0,ax]
   * @param nx Wave count x
   * @param E Scaling factor x
   */
  CavitySolution1D(
	  double mu,
	  double eps,
	  double current_time = 0.0,
	  double ax = 1,
	  double nx = 2,
	  double E = 1);

  /**
   * @brief Evaluates solution component at spatial point 
   * 
   * @param p Spatial point
   * @param component Solution component {0,1}
   * @return double
   */
  double value(
	  const dealii::Point<1> &p,
	  const unsigned int component = 0) const override;

  /**
   * @brief Evaluates solution at spatial point
   * 
   * @param p Spatial point
   * @param values Solution vector
   */
  void vector_value(
	  const dealii::Point<1> &p,
	  dealii::Vector<double> &values) const override;

 private:
  /**
  * Permittivity epsilon and permeability mu.
  *
  * Cavity solution only exists for constant mu and eps.
  */
  double mu;
  double eps;
  double speed_o_light;

  /**
   * Spatial extent of cavity solution.
   * 
   * The wave number is given by k.
   * 
   */
  double ax;
  double nx;
  double k;
  double E;
  double omega;
};

}// namespace Data
}// namespace MaxwellProblem1D

#endif// SOLUTIONS_1D_H
