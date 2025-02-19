#ifndef TIME_INTEGRATION_CRANKNICOLSON_H_
#define TIME_INTEGRATION_CRANKNICOLSON_H_

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "TimeIntegrator.h"
#include "VectorMassMatrix.h"

using namespace dealii;

namespace MaxwellProblem {
namespace TimeIntegration {


/**
*
* \brief Implementation of the crank nicolson scheme for linear Maxwells' equations
*/
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = Vector<double>>
class CrankNicolson : public TimeIntegrator<MassMatrixtype, CurlMatrixtype, Vectortype> {
 private:
  const MassMatrixtype &MH;
  const MassMatrixtype &ME;
  const MassMatrixtype &MH_inverse;
  const MassMatrixtype &ME_inverse;
  const CurlMatrixtype &CH;
  const CurlMatrixtype &CE;
  

  double timestep;

  dealii::SolverControl solver_control;
  dealii::SolverCG<MaxwellProblem::DataTypes::VectorMassMatrix> solver_cg;
  bool first_step = true;

  Vectortype rhs;
  Vectortype H_half;
  MaxwellProblem::DataTypes::VectorMassMatrix rhs_solv;
  MaxwellProblem::DataTypes::VectorMassMatrix E_solv;

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
  CrankNicolson(
    const MassMatrixtype &MH,
    const MassMatrixtype &ME,
	  const MassMatrixtype &MH_inv,
	  const MassMatrixtype &ME_inv,
	  const CurlMatrixtype &CH,
	  const CurlMatrixtype &CE,
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
	  Vectortype &H,
	  Vectortype &E,
	  Vectortype &j_current);

  /**
   * @brief Resets the inital step flag.
   * 
   * CN integrator reuses an old step. This routine forces him to recompute it.
   */
  void reset_initial_step();
};

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = Vector<double>>

/**
*
* \brief Implementation of the upwind crank nicolson scheme for linear Maxwells' equations
*/
class CrankNicolsonUpwind : public TimeIntegrator<MassMatrixtype, CurlMatrixtype, Vectortype> {
 private:
  const MassMatrixtype &MH_inverse;
  const MassMatrixtype &ME_inverse;
  const CurlMatrixtype &CH;
  const CurlMatrixtype &CE;
  const CurlMatrixtype &SH;
  const CurlMatrixtype &SE;
  

  double timestep;

  dealii::SolverControl solver_control;
  dealii::SolverGMRES<BlockVector<double>> solver;

  dealii::BlockVector<double> u;
  dealii::BlockVector<double> j_rhs;

  bool first_step = true;

 public:
  /**
   * Constructor
   * @param MH : Massmatrix of H-component
   * @param ME : Massmatrix of E-component
   * @param CH : Curlmatrix of H-component
   * @param CE : Curlmatrix of E-component
   * @param SH : Stabilization matrix for upwind fluxes
   * @param SE : Stabilization matrix for upwind fluxes
   * @param timestep : Timestep of integration scheme
   */
  CrankNicolsonUpwind(
	  const MassMatrixtype &MH_inv,
	  const MassMatrixtype &ME_inv,
	  const CurlMatrixtype &CH,
	  const CurlMatrixtype &CE,
    const CurlMatrixtype &SH,
    const CurlMatrixtype &SE,
	  double timestep);

  /**
   *
   * This routine computes one full-step of the scheme.
   * @param H : Previous H-component
   * @param E : Previous E-component
   * @param j_current : Right-hand side
   */
  void integrate_step(
	  Vectortype &H,
	  Vectortype &E,
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

#endif// TIME_INTEGRATION_CRANKNICOLSON_H_
