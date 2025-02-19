#ifndef TIME_INTEGRATION_LTS_LFC_HH_
#define TIME_INTEGRATION_LTS_LFC_HH_

#include "LTS_LFC.h"
#include "LocallyImplicitOperators.h"

namespace MaxwellProblem {
namespace TimeIntegration {

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::LTS_LFC(
	const MassMatrixtype &M,
	const MassMatrixtype &M_inv,
	const CurlMatrixtype &C,
	double timestep,
	unsigned int lfc_degree,
	double eta)
	: M(M),
	  M_inverse(M_inv),
	  C(C),
	  ops{C, M_inverse, M},
	  timestep(timestep),
	  lfc_degree(lfc_degree),
	  eta(eta),
	  initial_step(true) {
		compute_constants();
	  }

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &solution,
	Vectortype &j_current) {

	const auto timestep_half = timestep / 2;

	if (initial_step) {
		// init temporary vectors
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
		partial_E_tmp_filtered.reinit(partial_E_tmp);
		H_half.reinit(H_tmp);
		E_aux.reinit(E_tmp);
		rhs_solv.reinit(E_tmp);

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
		
		H_half = H_tmp + 0.5 * timestep * ops.mass_inv_H * ops.curl_E * E_tmp;
		initial_step = false;
	} else {
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
		H_half.sadd(-1., 2., H_tmp);
	}

	{
		rhs_solv.block(0).swap(j_current.block(4));
		rhs_solv.block(1).swap(j_current.block(5));
		rhs_solv.block(2).swap(j_current.block(6));
		rhs_solv.block(3).swap(j_current.block(7));
	}

	E_aux = ops.mass_inv_E * ops.curl_H * H_half + ops.mass_inv_E * rhs_solv;

	{
		partial_E_tmp.block(0).swap(E_aux.block(1));
		partial_E_tmp.block(1).swap(E_aux.block(2));
		partial_E_tmp.block(2).swap(E_aux.block(3));
	}

	compute_Pphat(partial_E_tmp_filtered, partial_E_tmp);

	{
		partial_E_tmp_filtered.block(0).swap(E_aux.block(1));
		partial_E_tmp_filtered.block(1).swap(E_aux.block(2));
		partial_E_tmp_filtered.block(2).swap(E_aux.block(3));
	}

	E_tmp.add(timestep, E_aux);

	H_tmp = H_half + timestep_half * ops.mass_inv_H * ops.curl_E * E_tmp;

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
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step_alternative(
	Vectortype &solution,
	Vectortype &j_current) {

	const auto timestep_half = timestep / 2;

	if (initial_step) {
		// init temporary vectors
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
		partial_H_tmp.reinit(
			{solution.block(1).size(),
			solution.block(2).size(),
			solution.block(3).size()});
		partial_E_tmp_filtered.reinit(partial_E_tmp);
		partial_H_tmp_filtered.reinit(partial_H_tmp);
		H_half.reinit(H_tmp);
		E_aux.reinit(E_tmp);
		rhs_solv.reinit(E_tmp);

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
		
		H_half = H_tmp + 0.5 * timestep * ops.mass_inv_H * ops.curl_E * E_tmp;
		initial_step = false;
	} else {
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
		H_half.sadd(-1., 2., H_tmp);
	}

	{
		rhs_solv.block(0).swap(j_current.block(4));
		rhs_solv.block(1).swap(j_current.block(5));
		rhs_solv.block(2).swap(j_current.block(6));
		rhs_solv.block(3).swap(j_current.block(7));
	}

	E_aux = ops.mass_inv_E * ops.curl_H * H_half + ops.mass_inv_E * rhs_solv;
	E_tmp.add(timestep, E_aux);
	E_aux.block(0) = 0.0;
	{
		partial_E_tmp.block(0).swap(E_aux.block(1));
		partial_E_tmp.block(1).swap(E_aux.block(2));
		partial_E_tmp.block(2).swap(E_aux.block(3));
	}
	partial_H_tmp = ops.sub_mass_inv_H * ops.sub_curl_E_imp * partial_E_tmp;

	compute_aux_Pphat(partial_H_tmp_filtered, partial_H_tmp);

	partial_E_tmp_filtered = ops.sub_mass_inv_E * ops.sub_curl_H_imp * partial_H_tmp_filtered;

	{
		partial_E_tmp_filtered.block(0).swap(E_aux.block(1));
		partial_E_tmp_filtered.block(1).swap(E_aux.block(2));
		partial_E_tmp_filtered.block(2).swap(E_aux.block(3));
	}

	E_tmp.add(-timestep*timestep*timestep, E_aux);

	H_tmp = H_half + timestep_half * ops.mass_inv_H * ops.curl_E * E_tmp;

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
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::compute_Pphat(
	Vectortype &vec_out,
	const Vectortype &vec_in) {

  if (lfc_degree == 1) {
	vec_out = vec_in;
  } else {
	// std::cout << alpha_p << " : " << stab_param << std::endl;
	tmp_cheb_rec1.reinit(vec_in);
	tmp_cheb_rec2 = vec_in;
	tmp_cheb_rec2 *= 2. / (alpha_p * stab_param);

	const auto filfun_op = - timestep * timestep * ops.sub_mass_inv_E * ops.sub_curl_H_imp * ops.sub_mass_inv_H * ops.sub_curl_E_imp;

	for (unsigned int k = 2; k <= lfc_degree; ++k) {
	  vec_out = filfun_op * tmp_cheb_rec2;
	  vec_out *= -1. / alpha_p;
	  vec_out.add(2. / alpha_p, vec_in);
	  vec_out.add(stab_param, tmp_cheb_rec2);
	  vec_out *= 2. * cheb_poly_stab[k-1] / cheb_poly_stab[k];
	  vec_out.add(-1. * cheb_poly_stab[k-2] / cheb_poly_stab[k], tmp_cheb_rec1);

	  if (k < lfc_degree) {
		tmp_cheb_rec1 = tmp_cheb_rec2;
		tmp_cheb_rec2 = vec_out;
	  }
	}
  }
}

/**
 * does not work so far?
 */
template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::compute_aux_Pphat(
	Vectortype &vec_out,
	const Vectortype &vec_in) {

  if (lfc_degree == 1) {
	vec_out = 0.0;
  } else {
	tmp_cheb_rec1.reinit(vec_in);
	tmp_cheb_rec2 = vec_in;
	tmp_cheb_rec2 *= -4. / (alpha_p*alpha_p * cheb_poly_stab[2]);
	if (lfc_degree == 2) vec_out = tmp_cheb_rec2;
	const auto filfun_op = -timestep * timestep * ops.sub_mass_inv_H * ops.sub_curl_E_imp * ops.sub_mass_inv_E * ops.sub_curl_H_imp;

	for (unsigned int k = 2; k <= lfc_degree - 1; ++k) {
	  vec_out = filfun_op * tmp_cheb_rec2;
	  vec_out *= -1. / alpha_p;
	  vec_out.add(-1. * alpha_k[k] / (alpha_p * alpha_p), vec_in);
	  vec_out.add(stab_param, tmp_cheb_rec2);
	  vec_out *= 2. * cheb_poly_stab[k] / cheb_poly_stab[k+1];
	  vec_out.add(-1. * cheb_poly_stab[k-1] / cheb_poly_stab[k+1], tmp_cheb_rec1);

	  if (k < lfc_degree - 1) {
		tmp_cheb_rec1 = tmp_cheb_rec2;
		tmp_cheb_rec2 = vec_out;
	  }
	}
  }
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::set_timestep(double timestep) {
  this->timestep = timestep;

  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LTS_LFC<MassMatrixtype, CurlMatrixtype, Vectortype>::compute_constants() {
  stab_param = 1. + eta * eta / (2. * lfc_degree * lfc_degree);

  std::vector<double> cheb_poly_second_stab(lfc_degree + 1);
  cheb_poly_stab.resize(lfc_degree + 1, 1.);
  alpha_k.resize(lfc_degree + 1, 2. * lfc_degree * lfc_degree);
  
  if (eta == 0.) {
	alpha_p = 2. * lfc_degree * lfc_degree;
  } else {
	cheb_poly_second_stab[0] = 1.;

	cheb_poly_stab[1] = stab_param;
	cheb_poly_second_stab[1] = 2. * stab_param;
	alpha_k[0] = 0.0;
	alpha_k[1] = 2. * cheb_poly_second_stab[0] / cheb_poly_stab[1];
	for (unsigned int k = 2; k <= lfc_degree; ++k) {
		cheb_poly_stab[k] = 2. * stab_param * cheb_poly_stab[k-1] - cheb_poly_stab[k-2];
		cheb_poly_second_stab[k] = 2. * stab_param * cheb_poly_second_stab[k-1] - cheb_poly_second_stab[k-2];
		alpha_k[k] = 2. * k * cheb_poly_second_stab[k-1] / cheb_poly_stab[k];
	}
	alpha_p = 2. * lfc_degree * cheb_poly_second_stab[lfc_degree - 1] / cheb_poly_stab[lfc_degree];
  }
}


}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif// TIME_INTEGRATION_LTS_LFC_HH_
