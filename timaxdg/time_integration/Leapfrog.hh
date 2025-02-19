#ifndef TIME_INTEGRATION_LEAPFROG_HH_
#define TIME_INTEGRATION_LEAPFROG_HH_

#include "Leapfrog.h"

namespace MaxwellProblem {
namespace TimeIntegration {

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
Leapfrog<MassMatrixtype, CurlMatrixtype, Vectortype>::Leapfrog(
	const MassMatrixtype &MH_inv,
	const MassMatrixtype &ME_inv,
	const CurlMatrixtype &CH,
	const CurlMatrixtype &CE,
	double timestep)
	: MH_inverse(&MH_inv),
	  ME_inverse(&ME_inv),
	  CH(&CH),
	  CE(&CE),
	  timestep(timestep),
	  initial_step(true) {}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void Leapfrog<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &H,
	Vectortype &E,
	Vectortype &j_current) {
  // First half step for H
  if (initial_step) {
	tmp_H.reinit(E);
	tmp_E.reinit(H);
	H_half.reinit(H);

	(*CE).vmult(tmp_E, E);
	(*MH_inverse).vmult(H_half, tmp_E);
	H_half.sadd(-1. * timestep / 2.0, H);

	initial_step = false;
  } else
	H_half.sadd(-1., 2., H);

  // Full step for E
  (*CH).vmult(tmp_H, H_half);
  tmp_H.sadd(timestep, -1. * timestep, j_current);
  (*ME_inverse).vmult_add(E, tmp_H);

  // Second half step for H
  (*CE).vmult(tmp_E, E);
  (*MH_inverse).vmult(H, tmp_E);
  H.sadd(-1. * timestep / 2.0, H_half);
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void Leapfrog<MassMatrixtype, CurlMatrixtype, Vectortype>::set_timestep(double timestep) {
  this->timestep = timestep;

  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void Leapfrog<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
LeapfrogUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::LeapfrogUpwind(
	const MassMatrixtype &MH_inv,
	const MassMatrixtype &ME_inv,
	const CurlMatrixtype &CH,
	const CurlMatrixtype &CE,
	double timestep,
	const CurlMatrixtype &SH,
	const CurlMatrixtype &SE)
	: MH_inverse(&MH_inv),
	  ME_inverse(&ME_inv),
	  CH(&CH),
	  CE(&CE),
	  SH(&SH),
	  SE(&SE),
	  timestep(timestep),
	  initial_step(true) {}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &H,
	Vectortype &E,
	Vectortype &j_current) {
  if (initial_step) {
	tmp_CE.reinit(H);
	tmp_CH.reinit(E);
	tmp_SH.reinit(H);

	(*CE).vmult(tmp_CE, E);

	initial_step = false;
  }

  // First half step for H
  (*SH).vmult(tmp_SH,
			  H);// tmp_SH is used at the second half step again.
  tmp_CE.sadd(-1. * timestep / 2.0, -1. * timestep / 2.0, tmp_SH);
  (*MH_inverse).vmult_add(H, tmp_CE);

  // Full step for E
  (*SE).vmult(tmp_CH, E);
  tmp_CH.sadd(-1.0, -1.0, j_current);
  (*CH).vmult_add(tmp_CH, H);
  tmp_CH *= timestep;
  (*ME_inverse).vmult_add(E, tmp_CH);

  // Second half step for H
  (*CE).vmult(tmp_CE,
			  E);// tmp_CE is computed and used in the next time step again.
  tmp_SH.sadd(-1. * timestep / 2.0, -1. * timestep / 2.0, tmp_CE);
  (*MH_inverse).vmult_add(H, tmp_SH);
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::set_timestep(double timestep) {
  this->timestep = timestep;

  initial_step = true;
}

template<typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LeapfrogUpwind<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step() {
  initial_step = true;
}

}// namespace TimeIntegration
}// namespace MaxwellProblem

#endif// TIME_INTEGRATION_LEAPFROG_HH_
