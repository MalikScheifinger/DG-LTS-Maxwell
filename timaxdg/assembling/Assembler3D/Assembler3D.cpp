#include "Assembler3D.h"

#include <vector>

#include "deal.II/dofs/dof_tools.h"
#include <deal.II/meshworker/mesh_loop.h>

#include "Assembler3DLocal.h"

namespace MaxwellProblem::Assembling {

Assembler3D::Assembler3D(
	dealii::FESystem<3> &fe,
	const dealii::MappingQ1<3> &mapping,
	const dealii::Quadrature<3> &quadrature,
	const dealii::Quadrature<2> &face_quadrature,
	dealii::DoFHandler<3> &dof_handler,
	dealii::Function<3> &mu_function,
	dealii::Function<3> &eps_function)
	: fe(fe),
	  mapping(mapping),
	  quadrature(quadrature),
	  face_quadrature(face_quadrature),
	  dof_handler(dof_handler),
	  mu_function(mu_function),
	  eps_function(eps_function),
	  fe_v(
		  mapping,
		  fe,
		  quadrature,
		  (dealii::UpdateFlags::update_values
		   | dealii::UpdateFlags::update_gradients
		   | dealii::UpdateFlags::update_quadrature_points
		   | dealii::UpdateFlags::update_JxW_values)),
	  fe_v_face(
		  mapping,
		  fe,
		  face_quadrature,
		  (dealii::UpdateFlags::update_values
		   | dealii::UpdateFlags::update_quadrature_points
		   | dealii::UpdateFlags::update_JxW_values
		   | dealii::UpdateFlags::update_normal_vectors)),
	  fe_v_subface(
		  mapping,
		  fe,
		  face_quadrature,
		  (dealii::UpdateFlags::update_values
		   | dealii::UpdateFlags::update_quadrature_points
		   | dealii::UpdateFlags::update_JxW_values
		   | dealii::UpdateFlags::update_normal_vectors)),
	  fe_v_face_neighbor(
		  mapping,
		  fe,
		  face_quadrature,
		  dealii::UpdateFlags::update_values) {}

void Assembler3D::generate_mass_pattern(
	dealii::BlockSparseMatrix<double> &mass_matrix,
	dealii::BlockSparsityPattern &mass_pattern) {

  std::vector<dealii::types::global_dof_index> dofs_per_block =
	  dealii::DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});

  const unsigned int n_H = dofs_per_block[0];
  const unsigned int n_E = dofs_per_block[1];

  dealii::BlockDynamicSparsityPattern dsp(2, 2);

  dsp.block(0, 0).reinit(n_H, n_H);
  dsp.block(1, 0).reinit(n_E, n_H);
  dsp.block(0, 1).reinit(n_H, n_E);
  dsp.block(1, 1).reinit(n_E, n_E);
  dsp.collect_sizes();

  const auto fe_components = fe.n_components();
  dealii::Table<2, dealii::DoFTools::Coupling> cell_coupling(fe_components, fe_components);

  for (auto const &i : {0, 1, 2}) {
	// coupling of H and H components
	cell_coupling[i][i] = dealii::DoFTools::always;
	for (auto const &j : {3, 4, 5}) {
	  // coupling of E and E components
	  cell_coupling[j][j] = dealii::DoFTools::always;
	  // coupling of H and E components
	  cell_coupling[i][j] = dealii::DoFTools::none;
	  // coupling of E and H components
	  cell_coupling[j][i] = dealii::DoFTools::none;
	}
  }

  dealii::DoFTools::make_sparsity_pattern(dof_handler, cell_coupling, dsp);
  mass_pattern.copy_from(dsp);
  mass_matrix.reinit(mass_pattern);
}

void Assembler3D::assemble_mass_matrix(dealii::BlockSparseMatrix<double> &mass_matrix) {
  //reset mass matrix
  mass_matrix = 0;

  const dealii::FEValuesExtractors::Vector H(0);
  auto dofs_per_cell_H = fe.get_sub_fe(fe.component_mask(H)).dofs_per_cell;

  std::vector<dealii::types::global_dof_index> dofs(fe.dofs_per_cell);

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell_H, dofs_per_cell_H);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
	cell_matrix = 0;
	fe_v.reinit(cell);

	assemble_mass_matrix_cell_H(cell_matrix, fe_v, dofs_per_cell_H);

	double eps = eps_function.value(cell->center());
	double mu = mu_function.value(cell->center());
	cell->get_dof_indices(dofs);
	for (unsigned int i = 0; i < dofs_per_cell_H; i++) {
	  for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
		mass_matrix.add(dofs[i], dofs[j], mu * cell_matrix(i, j));
		mass_matrix.add(dofs[i + dofs_per_cell_H], dofs[j + dofs_per_cell_H],
						eps * cell_matrix(j, i));
	  }
	}
  }
}

void Assembler3D::assemble_mass_matrix_parallel(dealii::BlockSparseMatrix<double> &mass_matrix) {
  // reset mass matrix
  mass_matrix = 0;

  MassScratchData3D mass_scratch_data(mapping, fe, quadrature, mu_function, eps_function);
  MassCopyData3D mass_copy_data;

  const dealii::FEValuesExtractors::Vector H(0);
  const auto dofs_per_cell_H = fe.get_sub_fe(fe.component_mask(H)).dofs_per_cell;

  const auto copier = [&](const MassCopyData3D &mass_copy_data) {
	for (unsigned int i = 0; i < dofs_per_cell_H; i++) {
	  for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
		mass_matrix.add(
			mass_copy_data.local_dof_indices[i],
			mass_copy_data.local_dof_indices[j],
			mass_copy_data.mu_value * mass_copy_data.cell_matrix(i, j));
		mass_matrix.add(
			mass_copy_data.local_dof_indices[i + dofs_per_cell_H],
			mass_copy_data.local_dof_indices[j + dofs_per_cell_H],
			mass_copy_data.eps_value * mass_copy_data.cell_matrix(j, i));
	  }
	}
  };

  dealii::MeshWorker::mesh_loop(
	  dof_handler.begin_active(),
	  dof_handler.end(),
	  mass_cell_worker_H,
	  copier,
	  mass_scratch_data,
	  mass_copy_data,
	  dealii::MeshWorker::assemble_own_cells);
}

void Assembler3D::assemble_mass_matrix(
	dealii::BlockSparseMatrix<double> &mass_matrix,
	dealii::BlockSparseMatrix<double> &mass_matrix_inv) {

  const dealii::FEValuesExtractors::Vector H(0);
  auto dofs_per_cell_H = fe.get_sub_fe(fe.component_mask(H)).dofs_per_cell;

  std::vector<dealii::types::global_dof_index> dofs(fe.dofs_per_cell);

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell_H, dofs_per_cell_H);
  dealii::FullMatrix<double> cell_matrix_inv(dofs_per_cell_H, dofs_per_cell_H);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
	cell_matrix = 0;
	cell_matrix_inv = 0;
	fe_v.reinit(cell);

	assemble_mass_matrix_cell_H(cell_matrix, fe_v, dofs_per_cell_H);

	cell_matrix_inv = cell_matrix;
	cell_matrix_inv.gauss_jordan();

	double eps = eps_function.value(cell->center());
	double mu = mu_function.value(cell->center());
	cell->get_dof_indices(dofs);
	for (unsigned int i = 0; i < dofs_per_cell_H; i++) {
	  for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
		mass_matrix.add(dofs[i], dofs[j], mu * cell_matrix(i, j));
		mass_matrix.add(dofs[i + dofs_per_cell_H], dofs[j + dofs_per_cell_H],
						eps * cell_matrix(j, i));
		mass_matrix_inv.add(dofs[i], dofs[j], cell_matrix_inv(i, j) / mu);
		mass_matrix_inv.add(dofs[i + dofs_per_cell_H], dofs[j + dofs_per_cell_H],
							cell_matrix_inv(j, i) / eps);
	  }
	}
  }
}

void Assembler3D::assemble_mass_matrix_parallel(
	dealii::BlockSparseMatrix<double> &mass_matrix,
	dealii::BlockSparseMatrix<double> &mass_matrix_inv) {
  // reset mass matrix
  mass_matrix = 0;
  mass_matrix_inv = 0;

  MassScratchData3D mass_scratch_data(mapping, fe, quadrature, mu_function, eps_function);
  MassCopyDataInv3D mass_copy_data;

  const dealii::FEValuesExtractors::Vector H(0);
  const auto dofs_per_cell_H = fe.get_sub_fe(fe.component_mask(H)).dofs_per_cell;

  const auto copier = [&](const MassCopyDataInv3D &mass_copy_data) {
	for (unsigned int i = 0; i < dofs_per_cell_H; i++) {
	  for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
		mass_matrix.add(
			mass_copy_data.local_dof_indices[i],
			mass_copy_data.local_dof_indices[j],
			mass_copy_data.mu_value * mass_copy_data.cell_matrix(i, j));
		mass_matrix.add(
			mass_copy_data.local_dof_indices[i + dofs_per_cell_H],
			mass_copy_data.local_dof_indices[j + dofs_per_cell_H],
			mass_copy_data.eps_value * mass_copy_data.cell_matrix(j, i));
		mass_matrix_inv.add(
			mass_copy_data.local_dof_indices[i],
			mass_copy_data.local_dof_indices[j],
			mass_copy_data.cell_matrix_inv(i, j) / mass_copy_data.mu_value);
		mass_matrix_inv.add(
			mass_copy_data.local_dof_indices[i + dofs_per_cell_H],
			mass_copy_data.local_dof_indices[j + dofs_per_cell_H],
			mass_copy_data.cell_matrix_inv(j, i) / mass_copy_data.eps_value);
	  }
	}
  };

  dealii::MeshWorker::mesh_loop(
	  dof_handler.begin_active(),
	  dof_handler.end(),
	  mass_cell_worker_inv_H,
	  copier,
	  mass_scratch_data,
	  mass_copy_data,
	  dealii::MeshWorker::assemble_own_cells);
}

void Assembler3D::generate_curl_pattern(
	dealii::BlockSparseMatrix<double> &curl_matrix,
	dealii::BlockSparsityPattern &curl_pattern) {

  std::vector<dealii::types::global_dof_index> dofs_per_block =
	  dealii::DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});

  const unsigned int n_H = dofs_per_block[0];
  const unsigned int n_E = dofs_per_block[1];

  dealii::BlockDynamicSparsityPattern dsp(2, 2);

  dsp.block(0, 0).reinit(n_H, n_H);
  dsp.block(1, 0).reinit(n_E, n_H);
  dsp.block(0, 1).reinit(n_H, n_E);
  dsp.block(1, 1).reinit(n_E, n_E);
  dsp.collect_sizes();

  const auto fe_components = fe.n_components();
  dealii::Table<2, dealii::DoFTools::Coupling> cell_coupling(
	  fe_components, fe_components);
  dealii::Table<2, dealii::DoFTools::Coupling> face_coupling(
	  fe_components, fe_components);

  for (auto const &i : {0, 1, 2}) {
	// coupling of H and H components
	cell_coupling[i][i] = dealii::DoFTools::none;
	face_coupling[i][i] = dealii::DoFTools::none;
	for (auto const &j : {3, 4, 5}) {
	  // coupling of E and E components
	  cell_coupling[j][j] = dealii::DoFTools::none;
	  face_coupling[j][j] = dealii::DoFTools::none;
	  // coupling of H and E
	  cell_coupling[i][j] = dealii::DoFTools::always;
	  face_coupling[i][j] = dealii::DoFTools::nonzero;
	  // coupling of E and H
	  cell_coupling[j][i] = dealii::DoFTools::always;
	  face_coupling[j][i] = dealii::DoFTools::nonzero;
	}
  }

  dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, cell_coupling, face_coupling);
  curl_pattern.copy_from(dsp);
  curl_matrix.reinit(curl_pattern);
}

void Assembler3D::assemble_curl_matrix(dealii::BlockSparseMatrix<double> &curl_matrix) {

  dealii::FEValuesExtractors::Vector H(0);
  dealii::FEValuesExtractors::Vector E(3);

  std::vector<dealii::types::global_dof_index> dofs(fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dofs_neighbor(fe.dofs_per_cell);

  auto dofs_per_cell_H = fe.get_sub_fe(fe.component_mask(H)).dofs_per_cell;
  auto dofs_per_cell_E = fe.get_sub_fe(fe.component_mask(E)).dofs_per_cell;

  dealii::FullMatrix<double> cell_curl_matrix(dofs_per_cell_E, dofs_per_cell_H);
  dealii::FullMatrix<double> cell_face_matrix_ext_int(dofs_per_cell_E, dofs_per_cell_H);
  dealii::FullMatrix<double> cell_face_matrix_int_ext(dofs_per_cell_E, dofs_per_cell_H);
  dealii::FullMatrix<double> cell_curl_matrix_ext(dofs_per_cell_E, dofs_per_cell_H);

  double eps, mu;
  double eps_neighbor, mu_neighbor;

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
	cell_curl_matrix = 0;
	fe_v.reinit(cell);

	eps = eps_function.value(cell->center());
	mu = mu_function.value(cell->center());

	assemble_cell_curl(cell_curl_matrix, fe_v, dofs_per_cell_E, dofs_per_cell_H);

	//Face Terms
	cell->get_dof_indices(dofs);
	for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<3>::faces_per_cell; ++face_no) {
	  auto face = cell->face(face_no);

	  // No boundary faces need to be considered since we only assemble C_E and there arent any
	  // boundary terms present.
	  if ((face->at_boundary())) continue;
	  Assert(cell->neighbor(face_no).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
	  auto neighbor = cell->neighbor(face_no);
	  if (face->has_children()) {
		const unsigned int neighbor2 = cell->neighbor_face_no(face_no);
		for (unsigned int subface_no = 0; subface_no < face->n_active_descendants(); ++subface_no) {
		  auto neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);
		  Assert(!neighbor_child->has_children(), dealii::ExcInternalError());

		  eps_neighbor = eps_function.value(neighbor_child->center());
		  mu_neighbor = mu_function.value(neighbor_child->center());

		  cell_face_matrix_ext_int = 0;
		  cell_face_matrix_int_ext = 0;
		  cell_curl_matrix_ext = 0;
		  fe_v_subface.reinit(cell, face_no, subface_no);
		  fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
		  assemble_face_curl(fe_v_subface,
							 fe_v_face_neighbor,
							 cell_curl_matrix,
							 cell_face_matrix_ext_int,
							 cell_face_matrix_int_ext,
							 cell_curl_matrix_ext,
							 eps,
							 mu,
							 eps_neighbor,
							 mu_neighbor);
		  neighbor_child->get_dof_indices(dofs_neighbor);

		  for (unsigned int i = 0; i < dofs_per_cell_E; i++) {
			for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
			  curl_matrix.add(dofs_neighbor[i + dofs_per_cell_H], dofs_neighbor[j],
							  cell_curl_matrix_ext(i, j));
			  curl_matrix.add(dofs_neighbor[j], dofs_neighbor[i + dofs_per_cell_H],
							  -cell_curl_matrix_ext(i, j));

			  curl_matrix.add(dofs_neighbor[i + dofs_per_cell_H], dofs[j],
							  cell_face_matrix_ext_int(i, j));
			  curl_matrix.add(dofs[j], dofs_neighbor[i + dofs_per_cell_H],
							  -cell_face_matrix_ext_int(i, j));

			  curl_matrix.add(dofs[i + dofs_per_cell_H], dofs_neighbor[j],
							  cell_face_matrix_int_ext(i, j));
			  curl_matrix.add(dofs_neighbor[j], dofs[i + dofs_per_cell_H],
							  -cell_face_matrix_int_ext(i, j));
			}
		  }
		}
	  } else {
		if (!cell->neighbor_is_coarser(face_no)
			&& (neighbor->index() > cell->index()
				|| (neighbor->level() < cell->level() && neighbor->index() == cell->index()))) {
		  const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

		  eps_neighbor = eps_function.value(neighbor->center());
		  mu_neighbor = mu_function.value(neighbor->center());

		  cell_face_matrix_ext_int = 0;
		  cell_face_matrix_int_ext = 0;
		  cell_curl_matrix_ext = 0;

		  fe_v_face.reinit(cell, face_no);
		  fe_v_face_neighbor.reinit(neighbor, neighbor2);

		  assemble_face_curl(fe_v_face,
							 fe_v_face_neighbor,
							 cell_curl_matrix,
							 cell_face_matrix_ext_int,
							 cell_face_matrix_int_ext,
							 cell_curl_matrix_ext,
							 eps,
							 mu,
							 eps_neighbor,
							 mu_neighbor);

		  neighbor->get_dof_indices(dofs_neighbor);

		  for (unsigned int i = 0; i < dofs_per_cell_E; i++) {
			for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
			  curl_matrix.add(dofs_neighbor[i + dofs_per_cell_H], dofs_neighbor[j],
							  cell_curl_matrix_ext(i, j));
			  curl_matrix.add(dofs_neighbor[j], dofs_neighbor[i + dofs_per_cell_H],
							  -cell_curl_matrix_ext(i, j));

			  curl_matrix.add(dofs_neighbor[i + dofs_per_cell_H], dofs[j],
							  cell_face_matrix_ext_int(i, j));
			  curl_matrix.add(dofs[j], dofs_neighbor[i + dofs_per_cell_H],
							  -cell_face_matrix_ext_int(i, j));

			  curl_matrix.add(dofs[i + dofs_per_cell_H], dofs_neighbor[j],
							  cell_face_matrix_int_ext(i, j));
			  curl_matrix.add(dofs_neighbor[j], dofs[i + dofs_per_cell_H],
							  -cell_face_matrix_int_ext(i, j));
			}
		  }
		}
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {

		curl_matrix.add(dofs[i + dofs_per_cell_H], dofs[j], cell_curl_matrix(i, j));
		curl_matrix.add(dofs[j], dofs[i + dofs_per_cell_H], -cell_curl_matrix(i, j));
	  }
	}
  }
}

void Assembler3D::assemble_curl_matrix_parallel(dealii::BlockSparseMatrix<double> &curl_matrix) {

  const auto copier = [&](const CurlCopyData3D &c) {
	if (!c.initialized) return;

	const auto dofs_per_cell_E = c.dofs_per_cell_E;
	const auto dofs_per_cell_H = c.dofs_per_cell_H;
	const auto &dofs = c.cell_dofs;

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {

		curl_matrix.add(dofs[i + dofs_per_cell_H], dofs[j], c.cell_matrix(i, j));
		curl_matrix.add(dofs[j], dofs[i + dofs_per_cell_H], -c.cell_matrix(i, j));
	  }
	}

	for (auto &cdf : c.face_data) {
	  const auto neighbor_dofs_per_cell_E = cdf.neighbor_dofs_per_cell_E;
	  const auto neighbor_dofs_per_cell_H = cdf.neighbor_dofs_per_cell_H;
	  const auto &dofs_neighbor = cdf.neighbor_cell_dofs;

	  for (unsigned int i = 0; i < neighbor_dofs_per_cell_E; i++) {
		for (unsigned int j = 0; j < neighbor_dofs_per_cell_H; j++) {
		  curl_matrix.add(dofs_neighbor[i + neighbor_dofs_per_cell_H], dofs_neighbor[j],
						  cdf.cell_matrix_ext(i, j));
		  curl_matrix.add(dofs_neighbor[j], dofs_neighbor[i + neighbor_dofs_per_cell_H],
						  -cdf.cell_matrix_ext(i, j));
		}
	  }
	  for (unsigned int i = 0; i < neighbor_dofs_per_cell_E; i++) {
		for (unsigned int j = 0; j < dofs_per_cell_H; j++) {
		  curl_matrix.add(dofs_neighbor[i + neighbor_dofs_per_cell_H], dofs[j],
						  cdf.face_matrix_ext_int(i, j));
		  curl_matrix.add(dofs[j], dofs_neighbor[i + neighbor_dofs_per_cell_H],
						  -cdf.face_matrix_ext_int(i, j));
		}
	  }
	  for (unsigned int i = 0; i < dofs_per_cell_E; i++) {
		for (unsigned int j = 0; j < neighbor_dofs_per_cell_H; j++) {
		  curl_matrix.add(dofs[i + dofs_per_cell_H], dofs_neighbor[j],
						  cdf.face_matrix_int_ext(i, j));
		  curl_matrix.add(dofs_neighbor[j], dofs[i + dofs_per_cell_H],
						  -cdf.face_matrix_int_ext(i, j));
		}
	  }
	}
  };

  CurlScratchData3D scratch_data(mapping, fe, quadrature, face_quadrature, mu_function, eps_function);
  CurlCopyData3D copy_data;
  /**
	 * Note that I could not figure out how to deal with anisotropic refinement in a
	 * interface worker. Because of this, all the interfaces are assembled in the cell worker.
	 * I assume this is slighlty less efficient.
	*/
  dealii::MeshWorker::mesh_loop(dof_handler.begin_active(),
								dof_handler.end(),
								curl_cell_worker,
								copier,
								scratch_data,
								copy_data,
								dealii::MeshWorker::assemble_own_cells,
								nullptr,// no boundary worker needed
								nullptr);
}

void Assembler3D::generate_stabilization_pattern(
	dealii::BlockSparseMatrix<double> &stab_matrix,
	dealii::BlockSparsityPattern &stab_pattern) {

  std::vector<dealii::types::global_dof_index> dofs_per_block =
	  dealii::DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});

  const unsigned int n_H = dofs_per_block[0];
  const unsigned int n_E = dofs_per_block[1];

  dealii::BlockDynamicSparsityPattern dsp(2, 2);

  dsp.block(0, 0).reinit(n_H, n_H);
  dsp.block(1, 0).reinit(n_E, n_H);
  dsp.block(0, 1).reinit(n_H, n_E);
  dsp.block(1, 1).reinit(n_E, n_E);
  dsp.collect_sizes();

  const auto fe_components = fe.n_components();
  dealii::Table<2, dealii::DoFTools::Coupling> cell_coupling(fe_components, fe_components);
  dealii::Table<2, dealii::DoFTools::Coupling> face_coupling(fe_components, fe_components);

  for (auto const &i : {0, 1, 2}) {
	// coupling of H and H components
	cell_coupling[i][i] = dealii::DoFTools::nonzero;
	face_coupling[i][i] = dealii::DoFTools::nonzero;
	for (auto const &j : {3, 4, 5}) {
	  // coupling of E and E components
	  cell_coupling[j][j] = dealii::DoFTools::nonzero;
	  face_coupling[j][j] = dealii::DoFTools::nonzero;
	  // coupling of H and E
	  cell_coupling[i][j] = dealii::DoFTools::none;
	  face_coupling[i][j] = dealii::DoFTools::none;
	  // coupling of E and H
	  cell_coupling[j][i] = dealii::DoFTools::none;
	  face_coupling[j][i] = dealii::DoFTools::none;
	}
  }

  // ToDo: Debug!
  //dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, cell_coupling, face_coupling);
  dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  stab_pattern.copy_from(dsp);
  stab_matrix.reinit(stab_pattern);
}

void Assembler3D::assemble_stabilization_matrix(
	dealii::BlockSparseMatrix<double> &stab_matrix,
	double alpha) {

  auto dofs_per_cell = fe.dofs_per_cell;

  std::vector<dealii::types::global_dof_index> dofs(fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dofs_neighbor(fe.dofs_per_cell);

  dealii::FullMatrix<double> cell_stab_matrix(dofs_per_cell, dofs_per_cell);
  dealii::FullMatrix<double> cell_face_matrix_ext_int(dofs_per_cell, dofs_per_cell);
  dealii::FullMatrix<double> cell_face_matrix_int_ext(dofs_per_cell, dofs_per_cell);
  dealii::FullMatrix<double> cell_stab_matrix_ext(dofs_per_cell, dofs_per_cell);

  double eps, mu;
  double eps_neighbor, mu_neighbor;

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
	cell_stab_matrix = 0;

	eps = eps_function.value(cell->center());
	mu = mu_function.value(cell->center());

	//Assemble Stab Term (consists only of face terms)
	cell->get_dof_indices(dofs);
	for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<3>::faces_per_cell; ++face_no) {
	  auto face = cell->face(face_no);

	  if (face->at_boundary()) {
		fe_v_face.reinit(cell, face_no);
		assemble_boundary_stab(fe_v_face, cell_stab_matrix, eps, mu);
	  } else {
		Assert(cell->neighbor(face_no).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
		auto neighbor = cell->neighbor(face_no);
		if (face->has_children()) {
		  const unsigned int neighbor2 = cell->neighbor_face_no(face_no);
		  for (unsigned int subface_no = 0; subface_no < face->n_active_descendants(); ++subface_no) {
			auto neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);
			Assert(!neighbor_child->has_children(), dealii::ExcInternalError());

			eps_neighbor = eps_function.value(neighbor_child->center());
			mu_neighbor = mu_function.value(neighbor_child->center());

			cell_face_matrix_ext_int = 0;
			cell_face_matrix_int_ext = 0;
			cell_stab_matrix_ext = 0;

			fe_v_subface.reinit(cell, face_no, subface_no);
			fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
			assemble_face_stab(fe_v_subface,
							   fe_v_face_neighbor,
							   cell_stab_matrix,
							   cell_face_matrix_ext_int,
							   cell_face_matrix_int_ext,
							   cell_stab_matrix_ext,
							   eps,
							   mu,
							   eps_neighbor,
							   mu_neighbor);
			neighbor_child->get_dof_indices(dofs_neighbor);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
				stab_matrix.add(dofs[i], dofs_neighbor[j], alpha * cell_face_matrix_int_ext(i, j));
				stab_matrix.add(dofs_neighbor[i], dofs[j], alpha * cell_face_matrix_ext_int(i, j));
				stab_matrix.add(dofs_neighbor[i], dofs_neighbor[j], alpha * cell_stab_matrix_ext(i, j));
			  }
			}
		  }
		} else {
		  if (!cell->neighbor_is_coarser(face_no) && (neighbor->index() > cell->index() || (neighbor->level() < cell->level() && neighbor->index() == cell->index()))) {
			const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

			eps_neighbor = eps_function.value(neighbor->center());
			mu_neighbor = mu_function.value(neighbor->center());

			cell_face_matrix_ext_int = 0;
			cell_face_matrix_int_ext = 0;
			cell_stab_matrix_ext = 0;

			fe_v_face.reinit(cell, face_no);
			fe_v_face_neighbor.reinit(neighbor, neighbor2);

			assemble_face_stab(fe_v_face,
							   fe_v_face_neighbor,
							   cell_stab_matrix,
							   cell_face_matrix_ext_int,
							   cell_face_matrix_int_ext,
							   cell_stab_matrix_ext,
							   eps,
							   mu,
							   eps_neighbor,
							   mu_neighbor);

			neighbor->get_dof_indices(dofs_neighbor);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
				stab_matrix.add(dofs[i], dofs_neighbor[j], alpha * cell_face_matrix_int_ext(i, j));
				stab_matrix.add(dofs_neighbor[i], dofs[j], alpha * cell_face_matrix_ext_int(i, j));
				stab_matrix.add(dofs_neighbor[i], dofs_neighbor[j], alpha * cell_stab_matrix_ext(i, j));
			  }
			}
		  }
		}
	  }
	}
	for (unsigned int i = 0; i < dofs_per_cell; ++i) {
	  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
		stab_matrix.add(dofs[i], dofs[j], alpha * cell_stab_matrix(i, j));
	  }
	}
  }
}

void Assembler3D::assemble_stabilization_matrix_parallel(
	dealii::BlockSparseMatrix<double> &stab_matrix,
	double alpha) {

  const auto copier = [&](const StabCopyData3D &c) {
	if (!c.initialized) return;

	const auto dofs_per_cell = c.dofs_per_cell;
	const auto &dofs = c.cell_dofs;

	for (unsigned int i = 0; i < dofs_per_cell; ++i) {
	  for (unsigned int j = 0; j < dofs_per_cell; ++j) {

		stab_matrix.add(dofs[i], dofs[j], alpha * c.cell_matrix(i, j));
	  }
	}

	for (auto &cdf : c.face_data) {
	  const auto neighbor_dofs_per_cell = cdf.neighbor_dofs_per_cell;
	  const auto &dofs_neighbor = cdf.neighbor_cell_dofs;

	  for (unsigned int i = 0; i < neighbor_dofs_per_cell; i++) {
		for (unsigned int j = 0; j < neighbor_dofs_per_cell; j++) {
		  stab_matrix.add(dofs_neighbor[i], dofs_neighbor[j], alpha * cdf.cell_matrix_ext(i, j));
		}
	  }
	  for (unsigned int i = 0; i < neighbor_dofs_per_cell; i++) {
		for (unsigned int j = 0; j < dofs_per_cell; j++) {
		  stab_matrix.add(dofs_neighbor[i], dofs[j], alpha * cdf.face_matrix_ext_int(i, j));
		}
	  }
	  for (unsigned int i = 0; i < dofs_per_cell; i++) {
		for (unsigned int j = 0; j < neighbor_dofs_per_cell; j++) {
		  stab_matrix.add(dofs[i], dofs_neighbor[j], alpha * cdf.face_matrix_int_ext(i, j));
		}
	  }
	}
  };

  StabScratchData3D scratch_data(mapping, fe, face_quadrature, mu_function, eps_function);
  StabCopyData3D copy_data;
  /**
	 * Note that I could not figure out how to deal with anisotropic refinement in an
	 * interface worker. Because of this, all the interfaces are assembled in the cell worker.
	 * I assume this is slighlty less efficient.
	*/
  dealii::MeshWorker::mesh_loop(dof_handler.begin_active(),
								dof_handler.end(),
								stab_cell_worker,
								copier,
								scratch_data,
								copy_data,
								dealii::MeshWorker::assemble_own_cells,
								nullptr,// no boundary worker needed
								nullptr);
}

void Assembler3D::assemble_rhs(
	const dealii::Function<3> &rhs_function,
	dealii::BlockVector<double> &rhs_vector) {

  rhs_vector = 0;

  const auto dofs_per_cell = fe_v.dofs_per_cell;

  std::vector<dealii::types::global_dof_index> dofs(dofs_per_cell);

  const unsigned int n_q_points = fe_v.n_quadrature_points;
  const unsigned int n_components_fe = fe.n_components();

  dealii::Vector<double> cell_vector(dofs_per_cell);
  std::vector<dealii::Vector<double>> rhs_values(
	  n_q_points, dealii::Vector<double>(n_components_fe));

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
	cell_vector = 0;
	fe_v.reinit(cell);

	const std::vector<double> &weights = fe_v.get_JxW_values();
	rhs_function.vector_value_list(fe_v.get_quadrature_points(), rhs_values);

	for (unsigned int point = 0; point < n_q_points; ++point)
	  for (unsigned int i = 0; i < dofs_per_cell; ++i) {
		const unsigned int component = fe.system_to_component_index(i).first;

		cell_vector(i) += rhs_values[point](component) * fe_v.shape_value(i, point) * weights[point];
	  }
	cell->get_dof_indices(dofs);

	for (unsigned int i = 0; i < dofs_per_cell; ++i)
	  rhs_vector(dofs[i]) += cell_vector(i);
  }
}

}// namespace MaxwellProblem::Assembling