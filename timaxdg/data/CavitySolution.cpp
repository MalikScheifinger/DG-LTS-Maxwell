#include <math.h>

#include "CavitySolution.h"

#include "deal.II/base/exceptions.h"
#include "deal.II/lac/vector.h"

namespace MaxwellProblem::Data {

CavitySolution3D::CavitySolution3D(
	double mu,
	double eps,
	double current_time,
	double ax,
	double ay,
	double az,
	double nx,
	double ny,
	double nz,
	double Ex,
	double Ey,
	double Ez)
	: dealii::Function<3, double>(6, current_time),
	  mu{mu},
	  eps{eps},
	  speed_o_light{1 / std::sqrt(mu * eps)},
	  ax{ax},
	  ay{ay},
	  az{az},
	  nx{nx},
	  ny{ny},
	  nz{nz},
	  kx{M_PI * nx / ax},
	  ky{M_PI * ny / ay},
	  kz{M_PI * nz / az},
	  Ex{Ex},
	  Ey{Ey},
	  Ez{Ez},
	  k{std::sqrt(kx * kx + ky * ky + kz * kz)},
	  omega{k * speed_o_light} {
  if (std::abs(kx * Ex + ky * Ey + kz * Ez) > 1e-6) {
	throw dealii::ExcMessage("Cavity Solution requirers kx*Ex + ky*Ey + kz*Ez = 0.");
  }
}

double CavitySolution3D::value(
	const dealii::Point<3> &p,
	const unsigned int component) const {

  const auto time = this->get_time();
  const auto sin_t = std::sin(omega * time);
  const auto cos_t = std::cos(omega * time);

  switch (component) {
	case 0:
	  return -(speed_o_light / k) * (Ez * ky - Ey * kz) * sin(kx * p[0]) * cos(ky * p[1]) * cos(kz * p[2]) * sin_t;
	case 1:
	  return -(speed_o_light / k) * (Ex * kz - Ez * kx) * cos(kx * p[0]) * sin(ky * p[1]) * cos(kz * p[2]) * sin_t;
	case 2:
	  return -(speed_o_light / k) * (Ey * kx - Ex * ky) * cos(kx * p[0]) * cos(ky * p[1]) * sin(kz * p[2]) * sin_t;
	case 3:
	  return (1 / eps) * Ex * cos(kx * p[0]) * sin(ky * p[1]) * sin(kz * p[2]) * cos_t;
	case 4:
	  return (1 / eps) * Ey * sin(kx * p[0]) * cos(ky * p[1]) * sin(kz * p[2]) * cos_t;
	case 5:
	  return (1 / eps) * Ez * sin(kx * p[0]) * sin(ky * p[1]) * cos(kz * p[2]) * cos_t;
	default:
	  throw dealii::ExcMessage("Cavity Solution only has 6 components.");
  }
}

void CavitySolution3D::vector_value(
	const dealii::Point<3> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto sin_t = sin(omega * time);
  const auto cos_t = cos(omega * time);

  values(0) = -(speed_o_light / k) * (Ez * ky - Ey * kz) * sin(kx * p[0]) * cos(ky * p[1]) * cos(kz * p[2]) * sin_t;
  values(1) = -(speed_o_light / k) * (Ex * kz - Ez * kx) * cos(kx * p[0]) * sin(ky * p[1]) * cos(kz * p[2]) * sin_t;
  values(2) = -(speed_o_light / k) * (Ey * kx - Ex * ky) * cos(kx * p[0]) * cos(ky * p[1]) * sin(kz * p[2]) * sin_t;
  values(3) = (1 / eps) * Ex * cos(kx * p[0]) * sin(ky * p[1]) * sin(kz * p[2]) * cos_t;
  values(4) = (1 / eps) * Ey * sin(kx * p[0]) * cos(ky * p[1]) * sin(kz * p[2]) * cos_t;
  values(5) = (1 / eps) * Ez * sin(kx * p[0]) * sin(ky * p[1]) * cos(kz * p[2]) * cos_t;
}

CavitySolutionTE::CavitySolutionTE(
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
	  speed_o_light{1 / std::sqrt(mu * eps)},
	  ax{ax},
	  ay{ay},
	  nx{nx},
	  ny{ny},
	  kx{M_PI * nx / ax},
	  ky{M_PI * ny / ay},
	  Ex{Ex},
	  Ey{Ey},
	  k{std::sqrt(kx * kx + ky * ky)},
	  omega{k * speed_o_light} {
  if (std::abs(kx * Ex + ky * Ey) > 1e-6) {
	throw dealii::ExcMessage("Cavity Solution requirers kx*Ex + ky*Ey = 0.");
  }
}

double CavitySolutionTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  const auto time = this->get_time();
  const auto sin_t = std::sin(omega * time);
  const auto cos_t = std::cos(omega * time);

  switch (component) {
	case 0:
	  return (speed_o_light / k) * (Ey * kx - Ex * ky) * cos(kx * p[0]) * cos(ky * p[1]) * sin_t;
	case 1:
	  return -(1 / eps) * Ex * cos(kx * p[0]) * sin(ky * p[1]) * cos_t;
	case 2:
	  return -(1 / eps) * Ey * sin(kx * p[0]) * cos(ky * p[1]) * cos_t;
	default:
	  throw dealii::ExcMessage("Cavity Solution only has 3 components.");
  }
}

void CavitySolutionTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto sin_t = sin(omega * time);
  const auto cos_t = cos(omega * time);

  values(0) =
	  (speed_o_light / k) * (Ey * kx - Ex * ky) * cos(kx * p[0]) * cos(ky * p[1]) * sin_t;
  values(1) = -(1 / eps) * Ex * cos(kx * p[0]) * sin(ky * p[1]) * cos_t;
  values(2) = -(1 / eps) * Ey * sin(kx * p[0]) * cos(ky * p[1]) * cos_t;
}

}// namespace MaxwellProblem::Data