#ifndef DATA_ISOTROPIC_CONSTANT_H_
#define DATA_ISOTROPIC_CONSTANT_H_

#include <deal.II/base/function.h>

namespace MaxwellProblem::Data {

/**
 * @brief Constant isotropic function.
 * 
 * This class provides an easy constant isotropic
 * parameter function used in Maxwell equations simulation.
 * 
 * @tparam dim Domain dimension.
 */
template<int dim>
class IsotropicConstant : public dealii::Function<dim> {
 public:

  /**
   * @brief Construct a new Isotropic Constant object
   * 
   * @param constant constant value the function takes.
   */
  IsotropicConstant(const double constant) : dealii::Function<dim>(1), constant(constant){};

  /**
   * @brief Returns constant
   * 
   * This function returns the constant.
   * It follows the calling convention of a dealii::Function
   * in order to be callable by library functions.
   */
  double value(const dealii::Point<dim> &/*p*/,
			   const unsigned int /*component = 0*/) const override {
	return constant;
  }

 private:
  const double constant;
};

}// namespace MaxwellProblem::Data

#endif//DATA_ISOTROPIC_CONSTANT_H_