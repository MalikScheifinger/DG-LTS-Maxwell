#ifndef ASSEMBLING_ASSEMBLER1D_LOCAL_H_
#define ASSEMBLING_ASSEMBLER1D_LOCAL_H_

#include "deal.II/fe/fe_values.h"
#include "deal.II/lac/full_matrix.h"
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>


namespace MaxwellProblem1D::Assembling {

// Serial

void assemble_mass_matrix_cell_H(
	dealii::FullMatrix<double> &cell_matrix,
	const dealii::FEValues<1> &fe_v,
	unsigned int dofs_per_cell_H);

void assemble_cell_curl(
	dealii::FullMatrix<double> &cell_curl_matrix,
	const dealii::FEValues<1> &fe_v,
	const double dofs_per_cell_E,
	const double dofs_per_cell_H);

void assemble_face_curl(
	const dealii::FEFaceValuesBase<1> &fe_v,
	const dealii::FEFaceValuesBase<1> &fe_v_neighbor,
	dealii::FullMatrix<double> &cell_curl_matrix,
	dealii::FullMatrix<double> &cell_face_matrix_ext_int,
	dealii::FullMatrix<double> &cell_face_matrix_int_ext,
	dealii::FullMatrix<double> &cell_curl_matrix_neighbor,
	double eps,
	double mu,
	double eps_neighbor,
	double mu_neighbor);

}// namespace MaxwellProblem1D::Assembling

#endif//ASSEMBLING_ASSEMBLER1D_LOCAL_H_