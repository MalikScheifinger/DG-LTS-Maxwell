#ifndef TIME_INTEGRATION_CRANKNICOLSON_HH_
#define TIME_INTEGRATION_CRANKNICOLSON_HH_

#include "CrankNicolson.h"

#include <deal.II/lac/precondition.h>

namespace MaxwellProblem {
namespace TimeIntegration {

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
CrankNicolson<MassMatrixtype, CurlMatrixtype, Vectortype>::CrankNicolson(
	const MassMatrixtype &MH,
	const MassMatrixtype &ME,
	const MassMatrixtype &MH_inv,
	const MassMatrixtype &ME_inv,
	const CurlMatrixtype &CH,
	const CurlMatrixtype &CE,
	double timestep,
  double solver_tol,
  unsigned int max_iters
  )
	: MH(MH),
	  ME(ME),
	  MH_inverse(MH_inv),
	  ME_inverse(ME_inv),
	  CH(CH),
	  CE(CE),
	  timestep(timestep),
	  solver_control(max_iters, solver_tol),
	  solver_cg(solver_control) {
  DataTypes::VectorMassMatrix::set_mass_matrix(ME);
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void CrankNicolson<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &H,
	Vectortype &E,
	Vectortype &j_current) {

  // L_cf = I + \tau^2 / 4 C_H C_E
  // L_cf E^{n+1} = b_E^n  + \tau / 2 C_H b_H^n
  // H^{n+1} = b_H^n -  \tau / 2 C_E E^{n+1}

  // Note: all the evaluations are lazy!
  // setup operators
  const auto MH_inv_op = dealii::linear_operator(MH_inverse);
  const auto ME_inv_op = dealii::linear_operator(ME_inverse);
  const auto CH_op = dealii::linear_operator(CH);
  const auto CE_op = dealii::linear_operator(CE);
  const auto I_E = dealii::identity_operator(ME_inv_op);

  const auto timestep_half = timestep / 2;
  const auto timestep_sqrt_quarter = (timestep * timestep) / 4;

  if (first_step) {
	rhs.reinit(E);
	H_half.reinit(H);

	rhs_solv.reinit(E);
	E_solv.reinit(E);

	H_half = timestep_half * MH_inv_op * CE_op * E;

	first_step = false;
  }

  // b steps
  const Vectortype b_H = H - H_half;
  rhs = E + timestep_half * ME_inv_op * CH_op * (H + b_H) + timestep * ME_inv_op * j_current;

  // schur complement
  const auto L_cf = I_E + timestep_sqrt_quarter * ME_inv_op * CH_op * MH_inv_op * CE_op;

  // E step

  E.swap(E_solv);
  rhs.swap(rhs_solv);

  solver_cg.solve(L_cf, E_solv, rhs_solv, dealii::PreconditionIdentity());

  E_solv.swap(E);

  H_half = timestep_half * MH_inv_op * CE_op * E;

  H = b_H - H_half;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void CrankNicolson<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  this->first_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
CrankNicolsonUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::CrankNicolsonUpwind(
	const MassMatrixtype &MH_inv,
	const MassMatrixtype &ME_inv,
	const CurlMatrixtype &CH,
	const CurlMatrixtype &CE,
	const CurlMatrixtype &SH,
	const CurlMatrixtype &SE,
	double timestep)
	: MH_inverse(MH_inv),
	  ME_inverse(ME_inv),
	  CH(CH),
	  CE(CE),
	  SH(SH),
	  SE(SE),
	  timestep(timestep),
	  solver_control(100, 1e-9),
	  solver(solver_control) {
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void CrankNicolsonUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &H,
	Vectortype &E,
	Vectortype &j_current) {

  // L_cf = I + \tau^2 / 4 C_H C_E
  // L_cf E^{n+1} = b_E^n  + \tau / 2 C_H b_H^n
  // H^{n+1} = b_H^n -  \tau / 2 C_E E^{n+1}

  const auto timestep_half = timestep / 2;

  // ops
  const auto MH_inv_op = dealii::linear_operator(MH_inverse);
  const auto ME_inv_op = dealii::linear_operator(ME_inverse);
  const auto CH_op = dealii::linear_operator(CH);
  const auto minus_CE_op = -1 * dealii::linear_operator(CE);
  const auto SH_op = dealii::linear_operator(SH);
  const auto SE_op = dealii::linear_operator(SE);
  const auto I_E = dealii::identity_operator(ME_inv_op);
  const auto I_H = dealii::identity_operator(MH_inv_op);

  // null ops
  const auto null_00 = dealii::null_operator(MH_inv_op);
  const auto null_01 = dealii::null_operator(minus_CE_op);
  const auto null_10 = dealii::null_operator(CH_op);
  const auto null_11 = dealii::null_operator(ME_inv_op);

  // block ops
  //block_operator<2, 2, BlockVector<double>>({op_a00, op_a01, op_a10, op_a11});
  const auto Mass_inverse_op = dealii::block_operator<2, 2, dealii::BlockVector<double>>({MH_inv_op, null_01, null_10, ME_inv_op});
  const auto Curl_op = dealii::block_operator<2, 2, dealii::BlockVector<double>>({null_00, minus_CE_op, CH_op, null_11});
  const auto Stab_op = dealii::block_operator<2, 2, dealii::BlockVector<double>>({SH_op, null_01, null_10, SE_op});
  const auto Identity_op = dealii::identity_operator(Curl_op);
  const auto R_plus = Identity_op + timestep_half * Mass_inverse_op * (Curl_op - Stab_op);
  const auto R_minus = Identity_op - timestep_half * Mass_inverse_op * (Curl_op - Stab_op);
  const auto R_minus_inverse = dealii::inverse_operator(R_minus, solver);

  if (first_step) {
    u.reinit(2);
    u.block(0).reinit(H);
    u.block(1).reinit(E);
    u.collect_sizes();
    u = 0;

    j_rhs.reinit(u);
    j_rhs = 0;
    first_step = false;
  }

  u.block(0).swap(H);
  u.block(1).swap(E);
  j_rhs.block(1).swap(j_current);
  
  // build rhs
  const auto rhs = R_plus * u + timestep * Mass_inverse_op * j_rhs; 
  
  // calculate step
  const auto u_step = R_minus_inverse * rhs;
  //solver.solve(R_minus, u, rhs, dealii::PreconditionIdentity());
  u_step.apply(u);
  
  u.block(0).swap(H);
  u.block(1).swap(E);
  j_rhs.block(1).swap(j_current);

}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void CrankNicolsonUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  this->first_step = true;
}

}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif//TIME_INTEGRATION_CRANKNICOLSON_HH_
