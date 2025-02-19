#include "VectorMassMatrix.h"

namespace MaxwellProblem {
namespace DataTypes {

const SparseMatrix<double> *VectorMassMatrix::mass_matrix = 0;

VectorMassMatrix::VectorMassMatrix()
	:
	Vector<double>() {}

VectorMassMatrix &VectorMassMatrix::operator=(const Vector<double> &v) {
  if(!mass_matrix) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassMatrixVectorMassMatrix without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix."
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  return (static_cast<VectorMassMatrix &>((*this).Vector<double>::operator=(v)));
}

VectorMassMatrix &VectorMassMatrix::operator=(const double s) {
  if(!mass_matrix) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassMatrixVectorMassMatrix without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix."
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  return (static_cast<VectorMassMatrix &>((*this).Vector<double>::operator=(s)));
}

double VectorMassMatrix::operator*(const VectorMassMatrix &v) const {
  Vector<double> tmp;
  tmp.reinit(v);

  if(!mass_matrix) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassMatrix without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix.\n"
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }

  mass_matrix->vmult(tmp, v);

  return (*this).Vector<double>::operator*(tmp);
}

double VectorMassMatrix::add_and_dot(
	const double a,
	const VectorMassMatrix &v,
	const VectorMassMatrix &w) {
  Vector<double> tmp;
  tmp.reinit(v);
  this->add(a, v);
  
  if(!mass_matrix) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassMatrix without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix.\n"
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  mass_matrix->vmult(tmp, w);

  return (*this).Vector<double>::operator*(tmp);
}

double VectorMassMatrix::l2_norm() const {
  if(!mass_matrix) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassMatrix without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix.\n"
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  return std::sqrt((*this) * (*this));
}

void VectorMassMatrix::set_mass_matrix(const SparseMatrix<double> &matrix) {
  mass_matrix = &matrix;
}

} /* namespace DataTypes */
} /* namespace MaxwellProblem */

#include <deal.II/lac/sparse_matrix.templates.h>
template void dealii::SparseMatrix<double>::vmult(MaxwellProblem::DataTypes::VectorMassMatrix &,
												  const MaxwellProblem::DataTypes::VectorMassMatrix &) const;

#include <deal.II/lac/vector_memory.templates.h>
template
class dealii::VectorMemory<MaxwellProblem::DataTypes::VectorMassMatrix>;
template
class dealii::GrowingVectorMemory<MaxwellProblem::DataTypes::VectorMassMatrix>;
