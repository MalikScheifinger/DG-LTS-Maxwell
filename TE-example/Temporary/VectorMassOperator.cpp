#include "VectorMassOperator.h"

namespace Temporary {
namespace DataTypes {

using Vectortype = typename dealii::BlockVector<double>;

using EmptyBlockPayload 
  = typename dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>;

const dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> *VectorMassOperator::mass_operator = 0;

VectorMassOperator::VectorMassOperator()
	:
	BlockVector<double>() {}

VectorMassOperator &VectorMassOperator::operator=(const BlockVector<double> &v) {
  if(!mass_operator) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassOperator without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix."
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  return (static_cast<VectorMassOperator &>((*this).BlockVector<double>::operator=(v)));
}

VectorMassOperator &VectorMassOperator::operator=(const double s) {
  if(!mass_operator) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassOperator without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix."
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  return (static_cast<VectorMassOperator &>((*this).BlockVector<double>::operator=(s)));
}

double VectorMassOperator::operator*(const VectorMassOperator &v) const {
  BlockVector<double> tmp;
  tmp.reinit(v);

  if(!mass_operator) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassOperator without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix.\n"
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }

  // mass_operator->vmult(tmp, v);

  mass_operator->vmult(tmp, v);

  return (*this).BlockVector<double>::operator*(tmp);
}

double VectorMassOperator::add_and_dot(
	const double a,
	const VectorMassOperator &v,
	const VectorMassOperator &w) {
  BlockVector<double> tmp;
  tmp.reinit(v);
  this->add(a, v);
  
  if(!mass_operator) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassOperator without\n"
      "setting the mass matrix.\n"
      "Note that the class has a static pointer to a mass matrix that"
      "that is used by it.\n"
      "Call the static function set_mass_matrix(&matrix)"
      "to set the matrix.\n"
      "Make also sure that the mass matrix pointed to wasn't freed in between!"
      );
  }
  mass_operator->vmult(tmp, w);

  return (*this).BlockVector<double>::operator*(tmp);
}

double VectorMassOperator::l2_norm() const {
  if(!mass_operator) {
    throw dealii::ExcMessage(
      "You probably tried to use the class VectorMassOperator without\n"
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

void VectorMassOperator::set_mass_operator(const dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> &op) {
  mass_operator = &op;
}

} /* namespace DataTypes */
} /* namespace Temporary */

#include <deal.II/lac/sparse_matrix.templates.h>
template void dealii::SparseMatrix<double>::vmult(Temporary::DataTypes::VectorMassOperator &,
												  const Temporary::DataTypes::VectorMassOperator &) const;

#include <deal.II/lac/vector_memory.templates.h>
template
class dealii::VectorMemory<Temporary::DataTypes::VectorMassOperator>;
template
class dealii::GrowingVectorMemory<Temporary::DataTypes::VectorMassOperator>;
