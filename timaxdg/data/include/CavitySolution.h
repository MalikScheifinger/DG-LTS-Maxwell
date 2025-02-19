#ifndef DATA_CAVITYSOLUTION_H_
#define DATA_CAVITYSOLUTION_H_

#include <deal.II/base/function.h>

namespace MaxwellProblem {
namespace Data {

/**
 * @brief Cavity solution for 3D mode.
 * 
 * This class implements en exact solution for maxwell equations
 * in 3D mode. A precise description can be found in 10.5445/IR/1000089271.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class CavitySolution3D : public dealii::Function<3, double> {
 public:
  /**
   * @brief Construct a new Cavity Solution 3D object
   *  
   * @param mu Permeability
   * @param eps Permittivity
   * @param current_time Time function gets initialized with
   * @param ax X-dimension parameter [0,ax]
   * @param ay Y-dimension parameter [0,ay]
   * @param az Z-dimension parameter [0,az]
   * @param nx Wave count x
   * @param ny Wave count y
   * @param nz Wave count z
   * @param Ex Scaling factor x
   * @param Ey Scaling factor y
   * @param Ez Scaling factor z
   */
  CavitySolution3D(
	  double mu,
	  double eps,
	  double current_time = 0.0,
	  double ax = 1,
	  double ay = 1,
	  double az = 1,
	  double nx = 2,
	  double ny = 2,
	  double nz = 2,
	  double Ex = -1,
	  double Ey = 0,
	  double Ez = 1);

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
   * The wave number is given by k = sqrt(kx*kx + ky*ky).
   * 
   * A condition that needs to be met is kx*Ex + ky*Ey = 0.
   */
  double ax;
  double ay;
  double az;
  double nx;
  double ny;
  double nz;
  double kx;
  double ky;
  double kz;
  double Ex;
  double Ey;
  double Ez;
  double k;
  double omega;
};

/**
 * @brief Cavity solution for RE mode.
 * 
 * This class implements en exact solution for maxwell equations
 * in RE mode.
 * 
 * The class inherits publicly from dealii::Function.
 * 
 */
class CavitySolutionTE : public dealii::Function<2, double> {
 public:
  /**
   * @brief Construct a new Cavity Solution TE object
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
  CavitySolutionTE(
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
   * The wave number is given by k = sqrt(kx*kx + ky*ky).
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
  double k;
  double omega;
};

}// namespace Data
}// namespace MaxwellProblem

#endif// DATA_CAVITYSOLUTION_H_
