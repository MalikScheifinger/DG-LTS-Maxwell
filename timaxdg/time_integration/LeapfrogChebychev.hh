#ifndef TIME_INTEGRATION_LFC_HH_
#define TIME_INTEGRATION_LFC_HH_

#include "LeapfrogChebychev.h"

namespace MaxwellProblem {
namespace TimeIntegration {

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
LeapfrogChebychev<MassMatrixtype, CurlMatrixtype, Vectortype>::LeapfrogChebychev(
	const MassMatrixtype &MH_inv,
	const MassMatrixtype &ME_inv,
	const CurlMatrixtype &CH,
	const CurlMatrixtype &CE,
	double timestep,
	unsigned int lfc_degree,
	double eta)
	: MH_inverse(MH_inv),
	  ME_inverse(ME_inv),
	  CH(CH),
	  CE(CE),
	  MH_inv_op(MH_inv),
	  ME_inv_op(ME_inv),
	  CH_op(CH),
	  CE_op(CE),
	  timestep(timestep),
	  lfc_degree(lfc_degree),
	  eta(eta),
	  initial_step(true) {
		compute_constants();
	  }

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogChebychev<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &H,
	Vectortype &E,
	Vectortype &j_current) {

	if (initial_step) {
		tmp_H.reinit(E);
		tmp_E_filtered.reinit(E);
		tmp_E.reinit(E);
		H_half.reinit(H);

		H_half = MH_inv_op * CE_op * E;
		H_half.sadd(timestep / 2.0, H);
		initial_step = false;
	} else {
		H_half.sadd(-1., 2., H);
	}

	tmp_H = CH_op * H_half;
	tmp_H.add(1., j_current);
	tmp_E = ME_inv_op * tmp_H;
	compute_Pphat(tmp_E_filtered, tmp_E);
	E.add(timestep, tmp_E_filtered);

	H = MH_inv_op * CE_op * E;
	H.sadd(timestep / 2.0, H_half);
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogChebychev<MassMatrixtype, CurlMatrixtype, Vectortype>::compute_Pphat(
	Vectortype &vec_out,
	const Vectortype &vec_in) {

  if (lfc_degree == 1) {
	vec_out = vec_in;
  } else {
	// std::cout << alpha_p << " : " << stab_param << std::endl;
	tmp_cheb_rec1.reinit(vec_in);
	tmp_cheb_rec2 = vec_in;
	tmp_cheb_rec2 *= 2. / (alpha_p * stab_param);

	const auto filfun_op = - timestep * timestep * ME_inv_op * CH_op * MH_inv_op * CE_op;

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

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogChebychev<MassMatrixtype, CurlMatrixtype, Vectortype>::set_timestep(double timestep) {
  this->timestep = timestep;

  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogChebychev<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogChebychev<MassMatrixtype, CurlMatrixtype, Vectortype>::compute_constants() {
  stab_param = 1. + eta * eta / (2. * lfc_degree * lfc_degree);

  std::vector<double> cheb_poly_second_stab(lfc_degree + 1);
  cheb_poly_stab.resize(lfc_degree + 1, 1.);
  
  if (eta == 0.) {
	alpha_p = 2. * lfc_degree * lfc_degree;
  } else {
	cheb_poly_second_stab[0] = 1.;

	cheb_poly_stab[1] = stab_param;
	cheb_poly_second_stab[1] = 2. * stab_param;

	for (unsigned int k = 2; k <= lfc_degree; ++k) {
		cheb_poly_stab[k] = 2. * stab_param * cheb_poly_stab[k-1] - cheb_poly_stab[k-2];
		cheb_poly_second_stab[k] = 2. * stab_param * cheb_poly_second_stab[k-1] - cheb_poly_second_stab[k-2];
	}

	alpha_p = 2. * lfc_degree * cheb_poly_second_stab[lfc_degree - 1] / cheb_poly_stab[lfc_degree];
  }
}


}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif// TIME_INTEGRATION_LFC_HH_
