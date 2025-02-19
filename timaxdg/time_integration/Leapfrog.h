#ifndef TIME_INTEGRATION_LEAPFROG_H_
#define TIME_INTEGRATION_LEAPFROG_H_

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include "TimeIntegrator.h"
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

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
* \brief Implementation of the leapfrog (or Verlet) scheme for linear Maxwells' equations
*/
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = Vector<double>>
class Leapfrog : public TimeIntegrator<MassMatrixtype, CurlMatrixtype, Vectortype> {
 public:
  /** \brief Constructor for Leapfrog scheme with central fluxes
   * @param MH_inv : Inverse Mass matrix of H-component
   * @param ME_inv : Inverse Mass matrix of E-component
   * @param CH : Curl matrix of H-component
   * @param CE : Curl matrix of E-component
   * @param timestep : timestep for integration scheme
   */
  Leapfrog(
	  const MassMatrixtype &MH_inv,
	  const MassMatrixtype &ME_inv,
	  const CurlMatrixtype &CH,
	  const CurlMatrixtype &CE,
	  double timestep);

  /**
   *
   * This routine computes one complete time step with the Leapfrog scheme.
   * @param H : Previous H-component
   * @param E : Previous E-component
   * @param j_current : Right-hand side
   */
  void integrate_step(
	  Vectortype &H,
	  Vectortype &E,
	  Vectortype &j_current);

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
   * Leapfrog integrator reuses an old step. This routine forces him to recompute it.
   */
  void reset_initial_step();

 private:
  const MassMatrixtype *MH_inverse;
  const MassMatrixtype *ME_inverse;
  const CurlMatrixtype *CH;
  const CurlMatrixtype *CE;

  double timestep;

  // temporary vectors for central flux
  Vectortype H_half;
  Vectortype tmp_H;
  Vectortype tmp_E;

  /**
   * Is true if initial step of Leapfrog scheme is computed.
   */
  bool initial_step;
};

/**
*
* \brief Implementation of the leapfrog (or Verlet) scheme for upwind linear Maxwells' equations
*/
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = Vector<double>>
class LeapfrogUpwind : public TimeIntegrator<MassMatrixtype, CurlMatrixtype, Vectortype> {
 public:
  /**
   * Constructor for Verlet scheme with upwind fluxes
   * @param MH_inv : Inverse Mass matrix of H-component
   * @param ME_inv : Inverse Mass matrix of E-component
   * @param CH : Curl matrix of H-component
   * @param CE : Curl matrix of E-component
   * @param timestep : timestep for integration scheme
   * @param SH : Stabilization matrix for upwind fluxes
   * @param SE : Stabilization matrix for upwind fluxes
   */
  LeapfrogUpwind(
	  const MassMatrixtype &MH_inv,
	  const MassMatrixtype &ME_inv,
	  const CurlMatrixtype &CH,
	  const CurlMatrixtype &CE,
	  double timestep,
	  const CurlMatrixtype &SH,
	  const CurlMatrixtype &SE);

  /**
   *
   * This routine computes one complete time step with the Leapfrog scheme.
   * @param H : Previous H-component
   * @param E : Previous E-component
   * @param j_current : Right-hand side
   */
  void integrate_step(
	  Vectortype &H,
	  Vectortype &E,
	  Vectortype &j_current);

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
   * Leapfrog integrator reuses an old step. This routine forces him to recompute it.
   */
  void reset_initial_step();

 private:
  const MassMatrixtype *MH_inverse;
  const MassMatrixtype *ME_inverse;
  const CurlMatrixtype *CH;
  const CurlMatrixtype *CE;

  const CurlMatrixtype *SH;
  const CurlMatrixtype *SE;

  double timestep;

  // temporary vectors for upwind flux
  Vectortype tmp_CE;
  Vectortype tmp_CH;
  Vectortype tmp_SH;

  /**
   * Is true if initial step of Leapfrog scheme is computed.
   */
  bool initial_step;
};

}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif// TIME_INTEGRATION_LEAPFROG_H_
