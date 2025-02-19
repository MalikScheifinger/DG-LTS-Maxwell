#include "NumberSpaces.h"

#include <cmath>

namespace MaxwellProblem::Tools {
std::vector<double> lin_spaced(const double start, const double stop, const int nums, const bool endpoint) {

  if (nums < 1) return std::vector<double>();

  double distance = stop - start;
  double width = distance / (nums - 1 + !endpoint);

  std::vector<double> points;
  points.resize(nums);

  points[0] = start;
  for (int i = 1; i < nums; i++) {
	points[i] = points[i - 1] + width;
  }

  return points;
}

std::vector<double> log_spaced(const double start, const double stop, const int nums, const bool endpoint) {

  if (nums < 1) return std::vector<double>();

  double distance = std::log(stop) - std::log(start);
  double width = distance / (nums-1 + !endpoint);

  std::vector<double> points;
  points.resize(nums);

  for (int i = 0; i < nums; i++) {
	points[i] = start * std::exp(width * i);
  }

  return points;
}

std::vector<double> log2_spaced(const double start, const double stop, const int nums, const bool endpoint) {

  if (nums < 1) return std::vector<double>();

  double distance = std::log2(stop) - std::log2(start);
  double width = distance / (nums-1 + !endpoint);

  std::vector<double> points;
  points.resize(nums);

  for (int i = 0; i < nums; i++) {
	points[i] = start * std::pow(2., width * i);
  }

  return points;
}

}// namespace MaxwellProblem::Tools