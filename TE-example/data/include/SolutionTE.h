#ifndef SOLUTION_TE_H_
#define SOLUTION_TE_H_

#include <deal.II/base/function.h>

/**
 * @brief Solution for TE mode with inhomogeneous right-hand side.
 * 
 * This class implements en exact solution for maxwell equations
 * in TE mode.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class SolutionTE : public dealii::Function<2, double> {
 public:
  /**
   * @brief Construct a new SolutionTE object
   *  
   * @param mu Permeability
   * @param eps Permittivity
   * @param current_time Time function gets initialized with
   * @param ax X-dimension parameter [0,ax]
   * @param ay Y-dimension parameter [0,ay]
   * @param nx Wave count x
   * @param ny Wave count y
   * @param Ex Scaling factor x
   * @param Ey Scaling factor y
   */
  SolutionTE(
	  double mu,
	  double eps,
	  double current_time = 0.0,
	  double ax = 1,
	  double ay = 1,
	  double nx = 2,
	  double ny = 2,
	  double Ex = -1,
	  double Ey = 1);

  /**
   * @brief Evaluates solution component at spatial point 
   * 
   * @param p Spatial point
   * @param component Solution component {0,1,2}
   * @return double
   */
  double value(
	  const dealii::Point<2> &p,
	  const unsigned int component = 0) const override;

  /**
   * @brief Evaluates solution at spatial point
   * 
   * @param p Spatial point
   * @param values Solution vector
   */
  void vector_value(
	  const dealii::Point<2> &p,
	  dealii::Vector<double> &values) const override;

 private:
  /**
  * Permittivity epsilon and permeability mu.
  */
  double mu;
  double eps;

  /**
   * Spatial extent of cavity solution.
   * 
   * The wave number is given by sqrt(kx*kx + ky*ky).
   * 
   * A condition that needs to be met is kx*Ex + ky*Ey = 0.
   */
  double ax;
  double ay;
  double nx;
  double ny;
  double kx;
  double ky;
  double Ex;
  double Ey;
};

/**
 * @brief Polynomial rhs for TE mode.
 * 
 * This class implements the rhs for SolutionTE for maxwell equations
 * in TE mode.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class RhsTE : public dealii::Function<2, double> {
 public:
  /**
   * @brief Construct a new Rhs TE object
   * 
   * @param mu Permeability
   * @param eps Permittivity
   * @param current_time Time function gets initialized with
   * @param ax X-dimension parameter [0,ax]
   * @param ay Y-dimension parameter [0,ay]
   * @param nx Wave count x
   * @param ny Wave count y
   * @param Ex Scaling factor x
   * @param Ey Scaling factor y
   */
  RhsTE(
	  double mu,
	  double eps,
	  double current_time = 0.0,
	  double ax = 1,
	  double ay = 1,
	  double nx = 2,
	  double ny = 2,
	  double Ex = -1,
	  double Ey = 1);

  /**
   * @brief Evaluates rhs component at spatial point 
   * 
   * @param p Spatial point
   * @param component Solution component {0,1,2}
   * @return double
   */
  double value(
	  const dealii::Point<2> &p,
	  const unsigned int component = 0) const override;

  /**
   * @brief Evaluates rhs at spatial point
   * 
   * @param p Spatial point
   * @param values Solution vector
   */
  void vector_value(
	  const dealii::Point<2> &p,
	  dealii::Vector<double> &values) const override;

  private:
  /**
  * Permittivity epsilon and permeability mu.
  */
  double mu;
  double eps;

  /**
   * A condition that needs to be met is kx*Ex + ky*Ey = 0.
   */
  double ax;
  double ay;
  double nx;
  double ny;
  double kx;
  double ky;
  double Ex;
  double Ey;
  double prefactor;
};

#endif// SOLUTION_TE_H_
