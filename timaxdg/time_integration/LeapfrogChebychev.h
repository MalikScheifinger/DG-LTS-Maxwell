#ifndef TIME_INTEGRATION_LFC_H_
#define TIME_INTEGRATION_LFC_H_

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include "TimeIntegrator.h"
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/linear_operator.h>

using namespace dealii;

namespace MaxwellProblem {

/**
 * \brief In this namespace all time integration routines are collected.
 *
 * These are at the moment the Verlet scheme, the Crank-Nicolson scheme and Krylov subspace methods.
 */
namespace TimeIntegration {

/**
*
* \brief Implementation of the leapfrog-Chebychev (LFC) scheme for linear Maxwells' equations
*/
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = Vector<double>>
class LeapfrogChebychev : public TimeIntegrator<MassMatrixtype, CurlMatrixtype, Vectortype> {
 public:
  /** \brief Constructor for LFC scheme with central fluxes
   * @param MH_inv : Inverse Mass matrix of H-component
   * @param ME_inv : Inverse Mass matrix of E-component
   * @param CH : Curl matrix of H-component
   * @param CE : Curl matrix of E-component
   * @param timestep : timestep for integration scheme
   */
  LeapfrogChebychev(
	  const MassMatrixtype &MH_inv,
	  const MassMatrixtype &ME_inv,
	  const CurlMatrixtype &CH,
	  const CurlMatrixtype &CE,
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
	  Vectortype &H,
	  Vectortype &E,
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
  const MassMatrixtype &MH_inverse;
  const MassMatrixtype &ME_inverse;
  const CurlMatrixtype &CH;
  const CurlMatrixtype &CE;

  LinearOperator<Vectortype> MH_inv_op;
  LinearOperator<Vectortype> ME_inv_op;
  LinearOperator<Vectortype> CH_op;
  LinearOperator<Vectortype> CE_op;

  double timestep;
  unsigned int lfc_degree;

  // LFC constants
  double eta;
  double stab_param;
  double alpha_p;
  std::vector<double> cheb_poly_stab;

  // temporary vectors for central flux
  Vectortype H_half;
  Vectortype tmp_H;
  Vectortype tmp_E;
  Vectortype tmp_E_filtered;

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

#endif// TIME_INTEGRATION_LFC_H_
