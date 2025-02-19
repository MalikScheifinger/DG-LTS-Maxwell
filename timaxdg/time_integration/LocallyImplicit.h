#ifndef TIME_INTEGRATION_LOCALLYIMPLICIT_H_
#define TIME_INTEGRATION_LOCALLYIMPLICIT_H_

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "VectorMassOperator.h"
#include "LocallyImplicitOperators.h"

using namespace dealii;

namespace MaxwellProblem {
namespace TimeIntegration {

/**
*
* \brief Implementation of the crank nicolson scheme for linear Maxwells' equations
*/
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = dealii::BlockVector<double>>
class LocallyImplicit {
 private:
  const MassMatrixtype &M;
  const MassMatrixtype &M_inverse;
  const CurlMatrixtype &C;

  double timestep;

  MaxwellProblem::LocallyImplicit::LocallyImplicitOperators ops;

  dealii::SolverControl solver_control;
  dealii::SolverCG<Temporary::DataTypes::VectorMassOperator> solver_cg;
  bool first_step = true;

  Vectortype H_tmp;
  Vectortype E_tmp;
  Temporary::DataTypes::VectorMassOperator partial_E_tmp;
  Vectortype H_half;
  Vectortype H_half_tmp;
  Vectortype rhs_solv;
  Temporary::DataTypes::VectorMassOperator partial_rhs_solv;


 public:
  /**
   * Constructor
   * @param MH : Massmatrix of H-component
   * @param ME : Massmatrix of E-component
   * @param MH_inv : Invers massmatrix of H-component
   * @param ME_inv : Invers massmatrix of E-component
   * @param CH : Curlmatrix of H-component
   * @param CE : Curlmatrix of E-component
   * @param timestep : Timestep of integration scheme
   */
  LocallyImplicit(
	  const MassMatrixtype &M,
	  const MassMatrixtype &M_inv,
	  const CurlMatrixtype &C,
	  double timestep,
    double solver_tol = 1e-12,
    unsigned int max_iters = 1000);

  /**
   *
   * This routine computes one full-step of the scheme.
   * @param H : Previous H-component
   * @param E : Previous E-component
   * @param j_current : Right-hand side
   */
  void integrate_step(
	  Vectortype &solution,
	  Vectortype &j_current);

  /**
   * @brief Resets the inital step flag.
   * 
   * CN integrator reuses an old step. This routine forces him to recompute it.
   */
  void reset_initial_step();
};

}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif// TIME_INTEGRATION_LOCALLYIMPLICIT_H_
