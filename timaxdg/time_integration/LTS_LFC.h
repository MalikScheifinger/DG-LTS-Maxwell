#ifndef TIME_INTEGRATION_LTS_LFC_H_
#define TIME_INTEGRATION_LTS_LFC_H_

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/linear_operator.h>

#include "TimeIntegrator.h"
#include "LocallyImplicitOperators.h"

using namespace dealii;

namespace MaxwellProblem {

/**
 * \brief In this namespace all time integration routines are collected.
 */
namespace TimeIntegration {

/**
*
* \brief Implementation of the leapfrog-Chebychev (LFC) local time-stepping (LTS) method for linear Maxwells' equations
*/
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = BlockVector<double>>
class LTS_LFC {
  
  public:
  /** \brief Constructor for LFC scheme with central fluxes
   * @param MH_inv : Inverse Mass matrix of H-component
   * @param ME_inv : Inverse Mass matrix of E-component
   * @param CH : Curl matrix of H-component
   * @param CE : Curl matrix of E-component
   * @param timestep : timestep for integration scheme
   */
  LTS_LFC(
	  const MassMatrixtype &M,
	  const MassMatrixtype &M_inv,
	  const CurlMatrixtype &C,
	  double timestep,
    unsigned int lfc_degree,
    double eta);

  /**
   *
   * This routine computes one complete time step with the LFC scheme.
   * @param H : Previous H-component
   * @param E : Previous E-component
   * @param j_current : Right-hand side
   */
  void integrate_step(
	  Vectortype &solution,
	  Vectortype &j_current);

    /**
   *
   * This routine computes one complete time step with the LFC scheme.
   * @param H : Previous H-component
   * @param E : Previous E-component
   * @param j_current : Right-hand side
   */
  void integrate_step_alternative(
	  Vectortype &solution,
	  Vectortype &j_current);

  /**
   * This routine computes the matrix-vector multiplication with the LFC function Pphat.
   * @param vec_out : Output vector
   * @param vec_in : Input vector
  */
  void compute_Pphat(
	  Vectortype &vec_out,
	  const Vectortype &vec_in);

  /**
   * This routine computes the matrix-vector multiplication with the LFC function Upsilon.
   * @param vec_out : Output vector
   * @param vec_in : Input vector
  */
  void compute_aux_Pphat(
	  Vectortype &vec_out,
	  const Vectortype &vec_in);

  /**
   * This routine changes the time step.
   * Only necessary if the time step is modified during time integration (or if several runs
   * with different time steps for the same space discretization are computed).
   * @param timestep : New timestep
   */
  void set_timestep(double timestep);

  /**
   * @brief Resets the inital step flag.
   * 
   * LFC integrator reuses an old step. This routine forces him to recompute it.
   */
  void reset_initial_step();

  /**
   * This routine computes the constants needed in every time step with the LFC scheme.
  */
  void compute_constants();

 private:
  const MassMatrixtype &M;
  const MassMatrixtype &M_inverse;
  const CurlMatrixtype &C;

  MaxwellProblem::LocallyImplicit::LocallyImplicitOperators ops;

  double timestep;
  unsigned int lfc_degree;

  // LFC constants
  double eta;
  double stab_param;
  double alpha_p;
  std::vector<double> alpha_k;
  std::vector<double> cheb_poly_stab;

  // temporary vectors
  Vectortype H_tmp;
  Vectortype E_tmp;
  Vectortype E_aux;
  Vectortype partial_E_tmp;
  Vectortype partial_E_tmp_filtered;
  Vectortype partial_H_tmp;
  Vectortype partial_H_tmp_filtered;
  Vectortype H_half;
  Vectortype rhs_solv;

  // temporary vectors for LFC recursion
  Vectortype tmp_cheb_rec1;
  Vectortype tmp_cheb_rec2;
  Vectortype cheb_rec_aux;

  /**
   * Is true if initial step of LFC scheme is computed.
   */
  bool initial_step;
};

}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif// TIME_INTEGRATION_LTS_LFC_H_
