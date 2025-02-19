#include <math.h>

#include "SolutionTE.h"

#include "deal.II/base/exceptions.h"
#include "deal.II/lac/vector.h"

SolutionTE::SolutionTE(
	double mu,
	double eps,
	double current_time,
	double ax,
	double ay,
	double nx,
	double ny,
	double Ex,
	double Ey)
	: dealii::Function<2, double>(3, current_time),
	  mu{mu},
	  eps{eps},
	  ax{ax},
	  ay{ay},
	  nx{nx},
	  ny{ny},
	  kx{M_PI * nx / ax},
	  ky{M_PI * ny / ay},
	  Ex{Ex},
	  Ey{Ey} {
  if (std::abs(kx * Ex + ky * Ey) > 1e-6) {
	throw dealii::ExcMessage("Solution requirers kx*Ex + ky*Ey = 0.");
  }
}

double SolutionTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  const auto time = this->get_time();
  const auto exp_t = exp(time);

  switch (component) {
	case 0:
	  return (1 / mu * eps) * (Ey * kx - Ex * ky) * cos(kx * p[0]) * cos(ky * p[1]) * exp_t;
	case 1:
	  return -(1 / eps) * Ex * cos(kx * p[0]) * sin(ky * p[1]) * exp_t;
	case 2:
	  return -(1 / eps) * Ey * sin(kx * p[0]) * cos(ky * p[1]) * exp_t;
	default:
	  throw dealii::ExcMessage("Cavity Solution only has 3 components.");
  }
}

void SolutionTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto exp_t = exp(time);

  values(0) = (1 / mu * eps) * (Ey * kx - Ex * ky) * cos(kx * p[0]) * cos(ky * p[1]) * exp_t;
  values(1) = -(1 / eps) * Ex * cos(kx * p[0]) * sin(ky * p[1]) * exp_t;
  values(2) = -(1 / eps) * Ey * sin(kx * p[0]) * cos(ky * p[1]) * exp_t;
}

RhsTE::RhsTE(double mu,
	double eps,
	double current_time,
	double ax,
	double ay,
	double nx,
	double ny,
	double Ex,
	double Ey)
	: dealii::Function<2, double>(3, current_time),
	  mu{mu},
	  eps{eps},
	  ax{ax},
	  ay{ay},
	  nx{nx},
	  ny{ny},
	  kx{M_PI * nx / ax},
	  ky{M_PI * ny / ay},
	  Ex{Ex},
	  Ey{Ey},
	  prefactor{(1 / mu * eps) * (Ey * kx - Ex * ky)} {}

double RhsTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  if (component == 0) return 0;

  const auto time = this->get_time();
  const auto exp_t = exp(time);

  switch (component) {
	case 1:
		{
			return (Ex - ky * prefactor) * cos(kx * p[0]) * sin(ky * p[1]) * exp_t;
		}
	case 2:
		{
			return (Ey + kx * prefactor) * sin(kx * p[0]) * cos(ky * p[1]) * exp_t;
		}
	default:
	  throw dealii::ExcMessage("Rhs only has 3 components.");
  }
}

void RhsTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto exp_t = exp(time);

  values(0) = 0;
  values(1) = (Ex - ky * prefactor) * cos(kx * p[0]) * sin(ky * p[1]) * exp_t;
  values(2) = (Ey + kx * prefactor) * sin(kx * p[0]) * cos(ky * p[1]) * exp_t;
}