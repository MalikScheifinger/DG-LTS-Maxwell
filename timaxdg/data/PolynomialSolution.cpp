#include "PolynomialSolution.h"

#include "deal.II/lac/vector.h"

/**
 * @brief Horner scheme to evaluate a polynomial.
 * 
 * The polynomial is in the form
 * p(x) = c[0] + c[1]*x + c[2]*x^2 + ... c[n]*x^n
 * where c is the coefficient vector.
 * 
 * @param coefficients Coefficient vector
 * @param x Point to evaluate at
 * @return double
 */
double inline horner_scheme(const std::vector<double> &coefficients, double x) {
  const int deg = coefficients.size() - 1;

  double ret = coefficients[deg];

  for (int idx = deg - 1; idx >= 0; idx--) {
	ret = ret * x + coefficients[idx];
  }

  return ret;
}

/**
 * @brief Horner scheme to evaluate the derivative of a polynomial.
 * 
 * The polynomial is in the form
 * p(x) = c[1]*x + 2*c[2]*x^1 + ... n*c[n]*x^(n-1)
 * where c is the coefficient vector.
 * 
 * @param coefficients Coefficient vector
 * @param x Point to evaluate at
 * @return double 
 */
double inline horner_scheme_derivative(
	const std::vector<double> &coefficients,
	double x) {
  const int deg = coefficients.size() - 1;

  double ret = deg * coefficients[deg];

  for (int idx = deg - 1; idx > 0; idx--) {
	ret = ret * x + idx * coefficients[idx];
  }

  return ret;
}

namespace MaxwellProblem::Data {

PolynomialSolution3D::PolynomialSolution3D(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<3, double>(6, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialSolution3D::value(
	const dealii::Point<3> &p,
	const unsigned int component) const {

  if (component == 0 || component == 1 || component == 2) return 0;

  const auto time = this->get_time();
  const auto time_polynomial = horner_scheme(time_coefficients, time);

  switch (component) {
	case 3:
		{
	  const auto space_0_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = horner_scheme(spatial_coefficients, p[2]);

	  return time_polynomial * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
		}
	case 4:
		{
	  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = horner_scheme(spatial_coefficients, p[2]);

	  return time_polynomial * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
		}
	case 5:
		{
	  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial_derivative = horner_scheme_derivative(spatial_coefficients, p[2]);

	  return time_polynomial * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 6 components.");
  }
}

void PolynomialSolution3D::vector_value(
	const dealii::Point<3> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial = horner_scheme(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[1]);
  const auto space_2_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[2]);
  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
  const auto space_2_polynomial = horner_scheme(spatial_coefficients, p[2]);

  values(0) = 0;
  values(1) = 0;
  values(2) = 0;
  values(3) = time_polynomial * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
  values(4) = time_polynomial * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
  values(5) = time_polynomial * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
}

PolynomialRhs3D::PolynomialRhs3D(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<3, double>(3, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialRhs3D::value(
	const dealii::Point<3> &p,
	const unsigned int component) const {

  if (component == 0 || component == 1 || component == 2) return 0;

  const auto time = this->get_time();
  const auto time_polynomial_derivative = horner_scheme_derivative(time_coefficients, time);

  switch (component) {
	case 3:
		{
	  const auto space_0_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = horner_scheme(spatial_coefficients, p[2]);

	  return - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
		}
	case 4:
		{
	  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = horner_scheme(spatial_coefficients, p[2]);

	  return - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
		}
	case 5:
		{
	  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial_derivative = horner_scheme_derivative(spatial_coefficients, p[2]);

	  return - time_polynomial_derivative * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 6 components.");
  }
}

void PolynomialRhs3D::vector_value(
	const dealii::Point<3> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial_derivative = horner_scheme_derivative(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[1]);
  const auto space_2_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[2]);
  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
  const auto space_2_polynomial = horner_scheme(spatial_coefficients, p[2]);

  values(0) = 0;
  values(1) = 0;
  values(2) = 0;
  values(3) = - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
  values(4) = - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
  values(5) = - time_polynomial_derivative * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
}

PolynomialSolutionTE::PolynomialSolutionTE(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<2, double>(3, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialSolutionTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  if (component == 0) return 0;

  const auto time = this->get_time();
  const auto time_polynomial = horner_scheme(time_coefficients, time);

  switch (component) {
	case 1:
		{
	  const auto space_0_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);

		return time_polynomial * space_0_polynomial_derivative * space_1_polynomial;
		}
	case 2:
		{
	  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[1]);

	  return time_polynomial * space_0_polynomial * space_1_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 3 components.");
  }
}

void PolynomialSolutionTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial = horner_scheme(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[1]);

  values(0) = 0;
  values(1) = time_polynomial * space_0_polynomial_derivative * space_1_polynomial;
  values(2) = time_polynomial * space_0_polynomial * space_1_polynomial_derivative;
}

PolynomialRhsTE::PolynomialRhsTE(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<2, double>(3, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialRhsTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  if (component == 0) return 0;

  const auto time = this->get_time();
  const auto time_polynomial_derivative = horner_scheme_derivative(time_coefficients, time);

  switch (component) {
	case 1:
		{
	  const auto space_0_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);

	  return - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial;
		}
	case 2:
		{
	  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  horner_scheme_derivative(spatial_coefficients, p[1]);

	  return - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 3 components.");
  }
}

void PolynomialRhsTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial_derivative = horner_scheme_derivative(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial = horner_scheme(spatial_coefficients, p[1]);
  const auto space_0_polynomial = horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  horner_scheme_derivative(spatial_coefficients, p[1]);

  values(0) = 0;
  values(1) = - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial;
  values(2) = - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative;
}

}// namespace MaxwellProblem::Data