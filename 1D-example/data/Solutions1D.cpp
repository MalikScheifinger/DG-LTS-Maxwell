#include <math.h>

#include "Solutions1D.h"

#include "deal.II/base/exceptions.h"
#include "deal.II/lac/vector.h"

namespace MaxwellProblem1D::Data {

CavitySolution1D::CavitySolution1D(
	double mu,
	double eps,
	double current_time,
	double ax,
	double nx,
	double E)
	: dealii::Function<1, double>(2, current_time),
	  mu{mu},
	  eps{eps},
	  speed_o_light{1 / std::sqrt(mu * eps)},
	  ax{ax},
	  nx{nx},
	  k{M_PI * nx / ax},
	  E{E},
	  omega{k * speed_o_light} {
}

double CavitySolution1D::value(
	const dealii::Point<1> &p,
	const unsigned int component) const {

  const auto time = this->get_time();
  const auto sin_t = std::sin(omega * time);
  const auto cos_t = std::cos(omega * time);

  switch (component) {
	case 0:
	  return -speed_o_light * E * cos(k * p[0]) * sin_t; // H
	case 1:
	  return 1 / eps * E * sin(k * p[0]) * cos_t; // E
	default:
	  throw dealii::ExcMessage("Cavity Solution only has 2 components.");
  }
}

void CavitySolution1D::vector_value(
	const dealii::Point<1> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto sin_t = sin(omega * time);
  const auto cos_t = cos(omega * time);

  values(0) = -speed_o_light * E * cos(k * p[0]) * sin_t; // H
  values(1) = 1 / eps * E * sin(k * p[0]) * cos_t; // E
}

}// namespace MaxwellProblem1D::Data