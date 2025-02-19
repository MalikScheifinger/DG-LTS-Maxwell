#ifndef DATA_TYPES_VECTOR_MASSMATRIX_H_
#define DATA_TYPES_VECTOR_MASSMATRIX_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/vector_memory.h>

using namespace dealii;

namespace MaxwellProblem {
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
class VectorMassMatrix : public Vector<double> {
 public:

  VectorMassMatrix();

  VectorMassMatrix &operator=(const Vector<double> &v);

  VectorMassMatrix &operator=(const double s);

  double operator*(const VectorMassMatrix &v) const;

  double add_and_dot(
	  const double a,
	  const VectorMassMatrix &v,
	  const VectorMassMatrix &w);

  double l2_norm() const;

  /**
   * @brief Set the mass matrix object
   * 
   * This method sets the static mass matrix pointer.
   * Make sure the matrix isn't freed away in between.
   * 
   * @param matrix 
   */
  static void set_mass_matrix(const SparseMatrix<double> &matrix);

 private:
  static const SparseMatrix<double> *mass_matrix;
};

} /* namespace DataTypes */
} /* namespace MaxwellProblem */


#endif /* DATA_TYPES_VECTOR_MASSMATRIX_H_ */
