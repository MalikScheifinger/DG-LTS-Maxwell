#ifndef DATA_POLSOLUTION_H
#define DATA_POLSOLUTION_H

#include <vector>

#include "deal.II/base/function.h"

namespace MaxwellProblem::Data {

/**
 * @brief Polynomial solution for 3D mode.
 * 
 * This class implements en exact solution for maxwell equations
 * in 3D mode if assembled with Polynomial3D
 * 
 * The polynomials are in the form
 * p(x) = c[1]*x + c[2]*x^2 + ... c[n]*x^n
 * where c is the coefficient vector.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class PolynomialSolution3D : public dealii::Function<3, double> {
 public:
  /**
   * @brief Construct a new Polynomial Solution 3D object
   * 
   * Note that with the given default spatial values the
   * solution fulfills the PEC boundary condition on the unit cube.
   * For different ax, ay and spatial_coefficients the user has
   * to guarantee that.
   * 
   * @param time_coefficients Time polynomial coefficients
   * @param current_time Time function gets initialized with
   * @param spatial_coefficients Spatial polynomial coefficients
   */
  PolynomialSolution3D(
	  std::vector<double> time_coefficients = {1, 1},
	  double current_time = 0.0,
	  std::vector<double> spatial_coefficients = {0, -1, 1});

  /**
   * @brief Evaluates solution component at spatial point 
   * 
   * @param p Spatial point
   * @param component Solution component {0,1,..,5}
   * @return double
   */
  double value(
	  const dealii::Point<3> &p,
	  const unsigned int component = 0) const override;

  /**
   * @brief Evaluates solution at spatial point
   * 
   * @param p Spatial point
   * @param values Solution vector
   */
  void vector_value(
	  const dealii::Point<3> &p,
	  dealii::Vector<double> &values) const override;

 private:
  std::vector<double> time_coefficients;
  std::vector<double> spatial_coefficients;
};

/**
 * @brief Polynomial rhs for 3D mode.
 * 
 * This class implements the rhs for PolynomialSolution3D for maxwell equations
 * in 3D mode.
 * 
 * The polynomials are in the form
 * p(x) = c[1]*x + c[2]*x^2 + ... c[n]*x^n
 * where c is the coefficient vector.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class PolynomialRhs3D : public dealii::Function<3, double> {
 public:
  /**
   * @brief Construct a new Polynomial Rhs 3D object
   * 
   * Note that with the given default spatial values the
   * solution fulfills the PEC boundary condition on the unit cube.
   * For different ax, ay and spatial_coefficients the user has
   * to guarantee that.
   * 
   * @param time_coefficients Time polynomial coefficients
   * @param current_time Time function gets initialized with
   * @param spatial_coefficients Spatial polynomial coefficients
   */
  PolynomialRhs3D(
	  std::vector<double> time_coefficients = {1, 1},
	  double current_time = 0.0,
	  std::vector<double> spatial_coefficients = {0, -1, 1});

  /**
   * @brief Evaluates rhs component at spatial point 
   * 
   * @param p Spatial point
   * @param component Solution component {0,1,..,5}
   * @return double
   */
  double value(
	  const dealii::Point<3> &p,
	  const unsigned int component = 0) const override;

  /**
   * @brief Evaluates rhs at spatial point
   * 
   * @param p Spatial point
   * @param values Solution vector
   */
  void vector_value(
	  const dealii::Point<3> &p,
	  dealii::Vector<double> &values) const override;

 private:
  std::vector<double> time_coefficients;
  std::vector<double> spatial_coefficients;
};

/**
 * @brief Polynomial solution for TE mode.
 * 
 * This class implements en exact solution for maxwell equations
 * in TE mode if assembled with PolynomialRhsTE
 * 
 * The polynomials are in the form
 * p(x) = c[1]*x + c[2]*x^2 + ... c[n]*x^n
 * where c is the coefficient vector.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class PolynomialSolutionTE : public dealii::Function<2, double> {
 public:
  /**
   * @brief Construct a new Polynomial Solution TE object
   * 
   * Note that with the given default spatial values the
   * solution fulfills the PEC boundary condition on the unit cube.
   * For different ax, ay and spatial_coefficients the user has
   * to guarantee that.
   * 
   * @param time_coefficients Time polynomial coefficients
   * @param current_time Time function gets initialized with
   * @param spatial_coefficients Spatial polynomial coefficients
   */
  PolynomialSolutionTE(
	  std::vector<double> time_coefficients = {1, 1},
	  double current_time = 0.0,
	  std::vector<double> spatial_coefficients = {0, -1, 1});

  /**
   * @brief Evaluates solution component at spatial point 
   * 
   * @param p Spatial point
   * @param component Solution component {0,1,3}
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
  std::vector<double> time_coefficients;
  std::vector<double> spatial_coefficients;
};

/**
 * @brief Polynomial rhs for TE mode.
 * 
 * This class implements the rhs for PolynomialSolutionTE for maxwell equations
 * in TE mode.
 * 
 * The polynomials are in the form
 * p(x) = c[1]*x + c[2]*x^2 + ... c[n]*x^n
 * where c is the coefficient vector.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class PolynomialRhsTE : public dealii::Function<2, double> {
 public:
  /**
   * @brief Construct a new Polynomial Rhs TE object
   * 
   * Note that with the given default spatial values the
   * solution fulfills the PEC boundary condition on the unit cube.
   * For different ax, ay and spatial_coefficients the user has
   * to guarantee that.
   * 
   * @param time_coefficients Time polynomial coefficients
   * @param current_time Time function gets initialized with
   * @param spatial_coefficients Spatial polynomial coefficients
   */
  PolynomialRhsTE(
	  std::vector<double> time_coefficients = {1, 1},
	  double current_time = 0.0,
	  std::vector<double> spatial_coefficients = {0, -1, 1});

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
  std::vector<double> time_coefficients;
  std::vector<double> spatial_coefficients;
};

}// namespace MaxwellProblem::Data

#endif// DATA_POLSOLUTION_H