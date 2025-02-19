#ifndef DATA_TYPES_VECTOR_MASSOPERATOR_TMP_H_
#define DATA_TYPES_VECTOR_MASSOPERATOR_TMP_H_

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/lac/vector_memory.h>

using namespace dealii;

namespace Temporary {
namespace DataTypes {

/**
 * @brief Vector type that allows for mass matrix weighted iterative solvers.
 * 
 * This vector type allows for iterative solvers to use an inner product that is 
 * weighted by a mass matrix.
 * 
 * This technic is for example used in the crank nicolson integrator. 
 * 
 * Please note that this class uses a static raw pointer to a matrix.
 * This pointer needs to be initialized before the class can be used.
 * Make sure you understand the class code before use since this
 * can introduce hard to debug errors!
 * 
 * Furthermore, only the necessary methods of a vector are implemented. 
 */
class VectorMassOperator : public BlockVector<double> {
 public:

  using Vectortype = typename dealii::BlockVector<double>;

  using EmptyBlockPayload 
    = typename dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>;

  VectorMassOperator();

  VectorMassOperator &operator=(const Vectortype &v);

  VectorMassOperator &operator=(const double s);

  double operator*(const VectorMassOperator &v) const;

  double add_and_dot(
	  const double a,
	  const VectorMassOperator &v,
	  const VectorMassOperator &w);

  double l2_norm() const;

  /**
   * @brief Set the mass matrix object
   * 
   * This method sets the static mass matrix pointer.
   * Make sure the matrix isn't freed away in between.
   * 
   * @param matrix 
   */
  static void set_mass_operator(const dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> &op);

 private:
  static const dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> *mass_operator;
};

} /* namespace DataTypes */
} /* namespace Temporary */

#endif /* DATA_TYPES_VECTOR_MASSOPERATOR_H_ */
