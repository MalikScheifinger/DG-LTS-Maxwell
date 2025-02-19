#ifndef ASSEMBLING_ASSEMBLER3D_LOCAL_H_
#define ASSEMBLING_ASSEMBLER3D_LOCAL_H_

#include "deal.II/fe/fe_values.h"
#include "deal.II/lac/full_matrix.h"
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>


namespace MaxwellProblem::Assembling {

// Serial

void assemble_mass_matrix_cell_H(
	dealii::FullMatrix<double> &cell_matrix,
	const dealii::FEValues<3> &fe_v,
	unsigned int dofs_per_cell_H);

void assemble_cell_curl(
	dealii::FullMatrix<double> &cell_curl_matrix,
	const dealii::FEValues<3> &fe_v,
	const double dofs_per_cell_E,
	const double dofs_per_cell_H);

void assemble_face_curl(
	const dealii::FEFaceValuesBase<3> &fe_v,
	const dealii::FEFaceValuesBase<3> &fe_v_neighbor,
	dealii::FullMatrix<double> &cell_curl_matrix,
	dealii::FullMatrix<double> &cell_face_matrix_ext_int,
	dealii::FullMatrix<double> &cell_face_matrix_int_ext,
	dealii::FullMatrix<double> &cell_curl_matrix_neighbor,
	double eps,
	double mu,
	double eps_neighbor,
	double mu_neighbor);

void assemble_face_stab(
	const dealii::FEFaceValuesBase<3> &fe_v,
	const dealii::FEFaceValuesBase<3> &fe_v_neighbor,
	dealii::FullMatrix<double> &cell_stab_matrix,
	dealii::FullMatrix<double> &cell_face_matrix_ext_int,
	dealii::FullMatrix<double> &cell_face_matrix_int_ext,
	dealii::FullMatrix<double> &cell_stab_matrix_neighbor,
	double eps,
	double mu,
	double eps_neighbor,
	double mu_neighbor);

void assemble_boundary_stab(
	const dealii::FEFaceValues<3> &fe_v,
	dealii::FullMatrix<double> &cell_stab_matrix_E,
	double eps,
	double mu);

// Parallel

struct MassScratchData3D {

  MassScratchData3D(
	  const dealii::Mapping<3> &mapping,
	  const dealii::FiniteElement<3> &fe,
	  const dealii::Quadrature<3> &quadrature,
	  const dealii::Function<3> &mu,
	  const dealii::Function<3> &eps,
	  const dealii::UpdateFlags update_flags =
		  (dealii::update_values
		   | dealii::update_quadrature_points
		   | dealii::update_JxW_values))
	  : fe_values(mapping, fe, quadrature, update_flags),
		mu(mu),
		eps(eps){};

  MassScratchData3D(const MassScratchData3D &scratch_data)
	  : fe_values(scratch_data.fe_values.get_mapping(),
				  scratch_data.fe_values.get_fe(),
				  scratch_data.fe_values.get_quadrature(),
				  scratch_data.fe_values.get_update_flags()),
		mu(scratch_data.mu),
		eps(scratch_data.eps){};

  dealii::FEValues<3> fe_values;
  const dealii::Function<3> &mu;
  const dealii::Function<3> &eps;
};

struct MassCopyData3D {
  dealii::FullMatrix<double> cell_matrix;
  std::vector<dealii::types::global_dof_index> local_dof_indices;
  double mu_value;
  double eps_value;

  template<class Iterator>
  void reinit(
	  const Iterator &cell,
	  unsigned int dofs_per_cell,
	  unsigned int dofs_per_cell_H,
	  double mu_value,
	  double eps_value) {

	cell_matrix.reinit(dofs_per_cell_H, dofs_per_cell_H);

	local_dof_indices.resize(dofs_per_cell);
	cell->get_dof_indices(local_dof_indices);

	this->mu_value = mu_value;
	this->eps_value = eps_value;
  }
};

struct MassCopyDataInv3D {
  dealii::FullMatrix<double> cell_matrix;
  dealii::FullMatrix<double> cell_matrix_inv;
  std::vector<dealii::types::global_dof_index> local_dof_indices;
  double mu_value;
  double eps_value;

  template<class Iterator>
  void reinit(
	  const Iterator &cell,
	  unsigned int dofs_per_cell,
	  unsigned int dofs_per_cell_H,
	  double mu_value,
	  double eps_value) {

	cell_matrix.reinit(dofs_per_cell_H, dofs_per_cell_H);
	cell_matrix_inv.reinit(dofs_per_cell_H, dofs_per_cell_H);

	local_dof_indices.resize(dofs_per_cell);
	cell->get_dof_indices(local_dof_indices);

	this->mu_value = mu_value;
	this->eps_value = eps_value;
  }
};

void mass_cell_worker_H(
	const dealii::DoFHandler<3>::active_cell_iterator &cell,
	MassScratchData3D &scratch_data,
	MassCopyData3D &copy_data);

void mass_cell_worker_inv_H(
	const dealii::DoFHandler<3>::active_cell_iterator &cell,
	MassScratchData3D &scratch_data,
	MassCopyDataInv3D &copy_data);

struct CurlScratchData3D {
  CurlScratchData3D(const dealii::Mapping<3> &mapping,
					const dealii::FiniteElement<3> &fe,
					const dealii::Quadrature<3> &quadrature,
					const dealii::Quadrature<2> &quadrature_face,
					const dealii::Function<3> &mu_function,
					const dealii::Function<3> &eps_function,
					const dealii::UpdateFlags update_flags =
						(dealii::update_values
						 | dealii::update_gradients
						 | dealii::update_quadrature_points
						 | dealii::update_JxW_values),
					const dealii::UpdateFlags face_update_flags =
						(dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors))
	  : fe_values(mapping, fe, quadrature, update_flags),
		fe_face_values(mapping, fe, quadrature_face, face_update_flags),
		fe_subface_values(mapping, fe, quadrature_face, face_update_flags),
		fe_face_values_neighbor(mapping, fe, quadrature_face, face_update_flags),
		mu_function(mu_function),
		eps_function(eps_function) {}

  CurlScratchData3D(const CurlScratchData3D &scratch_data)
	  : fe_values(
		  scratch_data.fe_values.get_mapping(),
		  scratch_data.fe_values.get_fe(),
		  scratch_data.fe_values.get_quadrature(),
		  scratch_data.fe_values.get_update_flags()),
		fe_face_values(
			scratch_data.fe_face_values.get_mapping(),
			scratch_data.fe_face_values.get_fe(),
			scratch_data.fe_face_values.get_quadrature(),
			scratch_data.fe_face_values.get_update_flags()),
		fe_subface_values(
			scratch_data.fe_subface_values.get_mapping(),
			scratch_data.fe_subface_values.get_fe(),
			scratch_data.fe_subface_values.get_quadrature(),
			scratch_data.fe_subface_values.get_update_flags()),
		fe_face_values_neighbor(
			scratch_data.fe_face_values_neighbor.get_mapping(),
			scratch_data.fe_face_values_neighbor.get_fe(),
			scratch_data.fe_face_values_neighbor.get_quadrature(),
			scratch_data.fe_face_values_neighbor.get_update_flags()),
		mu_function(scratch_data.mu_function),
		eps_function(scratch_data.eps_function) {}

  dealii::FEValues<3> fe_values;
  dealii::FEFaceValues<3> fe_face_values;
  dealii::FESubfaceValues<3> fe_subface_values;
  dealii::FEFaceValues<3> fe_face_values_neighbor;

  const dealii::Function<3> &mu_function;
  const dealii::Function<3> &eps_function;
};

struct CurlCopyFaceData3D {
  dealii::FullMatrix<double> cell_matrix_ext;
  dealii::FullMatrix<double> face_matrix_ext_int;
  dealii::FullMatrix<double> face_matrix_int_ext;

  unsigned int neighbor_dofs_per_cell_E = dealii::numbers::invalid_unsigned_int;
  unsigned int neighbor_dofs_per_cell_H = dealii::numbers::invalid_unsigned_int;

  std::vector<dealii::types::global_dof_index> neighbor_cell_dofs;

  template<class Iterator>
  void reinit(const Iterator &ncell,
			  unsigned int dofs_cell_E,
			  unsigned int dofs_cell_H,
			  unsigned int neighbor_dofs_cell_E,
			  unsigned int neighbor_dofs_cell_H) {

	neighbor_dofs_per_cell_E = neighbor_dofs_cell_E;
	neighbor_dofs_per_cell_H = neighbor_dofs_cell_H;

	cell_matrix_ext.reinit(neighbor_dofs_per_cell_E, neighbor_dofs_per_cell_H);
	face_matrix_ext_int.reinit(neighbor_dofs_per_cell_E, dofs_cell_H);
	face_matrix_int_ext.reinit(dofs_cell_E, neighbor_dofs_per_cell_H);

	neighbor_cell_dofs.resize(neighbor_dofs_per_cell_E + neighbor_dofs_per_cell_H);
	ncell->get_dof_indices(neighbor_cell_dofs);
  }
};

struct CurlCopyData3D {
  dealii::FullMatrix<double> cell_matrix;

  unsigned int dofs_per_cell_E = dealii::numbers::invalid_unsigned_int;
  unsigned int dofs_per_cell_H = dealii::numbers::invalid_unsigned_int;
  std::vector<dealii::types::global_dof_index> cell_dofs;

  std::vector<CurlCopyFaceData3D> face_data;

  bool initialized = false;

  template<class Iterator>
  void reinit(const Iterator &cell,
			  unsigned int dofs_cell_E,
			  unsigned int dofs_cell_H) {
	// check wether the copy data was already initilized.
	// cell worker and face worker use both the same
	// copy data.
	if (!initialized) {
	  dofs_per_cell_E = dofs_cell_E;
	  dofs_per_cell_H = dofs_cell_H;

	  cell_matrix.reinit(dofs_per_cell_E, dofs_per_cell_H);

	  cell_dofs.resize(dofs_per_cell_E + dofs_per_cell_H);
	  cell->get_dof_indices(cell_dofs);

	  initialized = true;
	}
  }
};

void curl_cell_worker(
	const dealii::DoFHandler<3>::active_cell_iterator &cell,
	CurlScratchData3D &scratch_data,
	CurlCopyData3D &copy_data);

struct StabScratchData3D {
  StabScratchData3D(const dealii::Mapping<3> &mapping,
					const dealii::FiniteElement<3> &fe,
					const dealii::Quadrature<2> &quadrature_face,
					const dealii::Function<3> &mu_function,
					const dealii::Function<3> &eps_function,
					const dealii::UpdateFlags face_update_flags =
						(dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors))
	  : fe_face_values(mapping, fe, quadrature_face, face_update_flags),
		fe_subface_values(mapping, fe, quadrature_face, face_update_flags),
		fe_face_values_neighbor(mapping, fe, quadrature_face, face_update_flags),
		mu_function(mu_function),
		eps_function(eps_function) {}

  StabScratchData3D(const StabScratchData3D &scratch_data)
	  : fe_face_values(
		  scratch_data.fe_face_values.get_mapping(),
		  scratch_data.fe_face_values.get_fe(),
		  scratch_data.fe_face_values.get_quadrature(),
		  scratch_data.fe_face_values.get_update_flags()),
		fe_subface_values(
			scratch_data.fe_subface_values.get_mapping(),
			scratch_data.fe_subface_values.get_fe(),
			scratch_data.fe_subface_values.get_quadrature(),
			scratch_data.fe_subface_values.get_update_flags()),
		fe_face_values_neighbor(
			scratch_data.fe_face_values_neighbor.get_mapping(),
			scratch_data.fe_face_values_neighbor.get_fe(),
			scratch_data.fe_face_values_neighbor.get_quadrature(),
			scratch_data.fe_face_values_neighbor.get_update_flags()),
		mu_function(scratch_data.mu_function),
		eps_function(scratch_data.eps_function) {}

  dealii::FEFaceValues<3> fe_face_values;
  dealii::FESubfaceValues<3> fe_subface_values;
  dealii::FEFaceValues<3> fe_face_values_neighbor;

  const dealii::Function<3> &mu_function;
  const dealii::Function<3> &eps_function;
};

struct StabCopyFaceData3D {
  dealii::FullMatrix<double> cell_matrix_ext;
  dealii::FullMatrix<double> face_matrix_ext_int;
  dealii::FullMatrix<double> face_matrix_int_ext;

  unsigned int neighbor_dofs_per_cell = dealii::numbers::invalid_unsigned_int;

  std::vector<dealii::types::global_dof_index> neighbor_cell_dofs;

  template<class Iterator>
  void reinit(const Iterator &ncell,
			  unsigned int dofs_per_cell,
			  unsigned int neighbor_dofs_per_cell) {

	this->neighbor_dofs_per_cell = neighbor_dofs_per_cell;

	cell_matrix_ext.reinit(neighbor_dofs_per_cell, neighbor_dofs_per_cell);
	face_matrix_ext_int.reinit(neighbor_dofs_per_cell, dofs_per_cell);
	face_matrix_int_ext.reinit(dofs_per_cell, neighbor_dofs_per_cell);

	neighbor_cell_dofs.resize(neighbor_dofs_per_cell);
	ncell->get_dof_indices(neighbor_cell_dofs);
  }
};

struct StabCopyData3D {
  dealii::FullMatrix<double> cell_matrix;

  unsigned int dofs_per_cell = dealii::numbers::invalid_unsigned_int;
  std::vector<dealii::types::global_dof_index> cell_dofs;

  std::vector<StabCopyFaceData3D> face_data;

  bool initialized = false;

  template<class Iterator>
  void reinit(const Iterator &cell,
			  unsigned int dofs_cell) {
	// check wether the copy data was already initilized.
	// cell worker and face worker use both the same
	// copy data.
	if (!initialized) {
	  dofs_per_cell = dofs_cell;

	  cell_matrix.reinit(dofs_per_cell, dofs_per_cell);

	  cell_dofs.resize(dofs_per_cell);
	  cell->get_dof_indices(cell_dofs);

	  initialized = true;
	}
  }
};

void stab_cell_worker(
	const dealii::DoFHandler<3>::active_cell_iterator &cell,
	StabScratchData3D &scratch_data,
	StabCopyData3D &copy_data);

}// namespace MaxwellProblem::Assembling

#endif//ASSEMBLING_ASSEMBLER3D_LOCAL_H_