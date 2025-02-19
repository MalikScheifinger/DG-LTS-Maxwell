#ifndef TIME_INTEGRATION_LOCALLYIMPLICIT_HH_
#define TIME_INTEGRATION_LOCALLYIMPLICIT_HH_

#include <utility>
#include <array>

#include "LocallyImplicit.h"
#include "LocallyImplicitOperators.h"

#include <deal.II/lac/precondition.h>

namespace MaxwellProblem {
namespace TimeIntegration {

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
LocallyImplicit<MassMatrixtype, CurlMatrixtype, Vectortype>::LocallyImplicit(
	const MassMatrixtype &M,
	const MassMatrixtype &M_inv,
	const CurlMatrixtype &C,
	double timestep,
	double solver_tol,
	unsigned int max_iters)
	: M(M),
	  M_inverse(M_inv),
	  C(C),
	  timestep(timestep),
    ops{C, M_inverse, M},
	  solver_control(max_iters, solver_tol),
	  solver_cg(solver_control) {
	Temporary::DataTypes::VectorMassOperator::set_mass_operator(ops.sub_mass_E);
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LocallyImplicit<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &solution,
	Vectortype &j_current) {

  const auto timestep_half = timestep / 2;
  const auto timestep_sqrt_quarter = (timestep * timestep) / 4;

  // schur complement
  const auto L_schur = dealii::identity_operator(ops.sub_mass_inv_E) 
    - timestep_sqrt_quarter * ops.sub_mass_inv_E * ops.sub_curl_H_imp * ops.sub_mass_inv_H * ops.sub_curl_E_imp;

  if (first_step) {
	// init all tmp vectors
	H_tmp.reinit(
		{solution.block(0).size(),
		 solution.block(1).size(),
		 solution.block(2).size(),
		 solution.block(3).size()});
	E_tmp.reinit(
		{solution.block(4).size(),
		 solution.block(5).size(),
		 solution.block(6).size(),
     	 solution.block(7).size()});
  	partial_E_tmp.reinit(
    	{solution.block(5).size(),
		 solution.block(6).size(),
     	 solution.block(7).size()});
  
  H_half.reinit(H_tmp);
  H_half_tmp.reinit(H_tmp);
  rhs_solv.reinit(E_tmp);
  partial_rhs_solv.reinit(partial_E_tmp);
	{
	  H_tmp.block(0).swap(solution.block(0));
	  H_tmp.block(1).swap(solution.block(1));
	  H_tmp.block(2).swap(solution.block(2));
	  H_tmp.block(3).swap(solution.block(3));
	  E_tmp.block(0).swap(solution.block(4));
	  E_tmp.block(1).swap(solution.block(5));
	  E_tmp.block(2).swap(solution.block(6));
	  E_tmp.block(3).swap(solution.block(7));
	}

	H_half_tmp = timestep_half * ops.mass_inv_H * ops.curl_E * E_tmp;

	first_step = false;
  } else {
	//swap solution
	{
	  H_tmp.block(0).swap(solution.block(0));
	  H_tmp.block(1).swap(solution.block(1));
	  H_tmp.block(2).swap(solution.block(2));
	  H_tmp.block(3).swap(solution.block(3));
	  E_tmp.block(0).swap(solution.block(4));
	  E_tmp.block(1).swap(solution.block(5));
	  E_tmp.block(2).swap(solution.block(6));
	  E_tmp.block(3).swap(solution.block(7));
	}
  }

	H_half = H_tmp + H_half_tmp;

  	rhs_solv = E_tmp 
		+ timestep * ops.mass_inv_E * ops.curl_H_exp * H_half 
		+ timestep_half * ops.mass_inv_E * ops.curl_H_imp * (H_half + H_tmp)
		- timestep * ops.mass_inv_E * j_current;

  // we only need to solve the implicit part
  // swap vectors
  {
    partial_E_tmp.block(0).swap(E_tmp.block(1));
    partial_E_tmp.block(1).swap(E_tmp.block(2));
    partial_E_tmp.block(2).swap(E_tmp.block(3));
    partial_rhs_solv.block(0).swap(rhs_solv.block(1));
    partial_rhs_solv.block(1).swap(rhs_solv.block(2));
    partial_rhs_solv.block(2).swap(rhs_solv.block(3));
  }

  solver_cg.solve(L_schur, partial_E_tmp, partial_rhs_solv, dealii::PreconditionIdentity());

  // swap vectors
  {
    partial_E_tmp.block(0).swap(E_tmp.block(1));
    partial_E_tmp.block(1).swap(E_tmp.block(2));
    partial_E_tmp.block(2).swap(E_tmp.block(3));
    partial_rhs_solv.block(0).swap(rhs_solv.block(1));
    partial_rhs_solv.block(1).swap(rhs_solv.block(2));
    partial_rhs_solv.block(2).swap(rhs_solv.block(3));
    // swap rhs with solution for explicit part
    E_tmp.block(0).swap(rhs_solv.block(0));
  }

  H_half_tmp = timestep_half * ops.mass_inv_H * ops.curl_E * E_tmp;
  
  H_tmp = H_half + H_half_tmp;

  //swap solution
	{
	  H_tmp.block(0).swap(solution.block(0));
	  H_tmp.block(1).swap(solution.block(1));
	  H_tmp.block(2).swap(solution.block(2));
	  H_tmp.block(3).swap(solution.block(3));
	  E_tmp.block(0).swap(solution.block(4));
	  E_tmp.block(1).swap(solution.block(5));
	  E_tmp.block(2).swap(solution.block(6));
	  E_tmp.block(3).swap(solution.block(7));
	}
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LocallyImplicit<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  this->first_step = true;
}

}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif//TIME_INTEGRATION_LOCALLYIMPLICIT_HH_
