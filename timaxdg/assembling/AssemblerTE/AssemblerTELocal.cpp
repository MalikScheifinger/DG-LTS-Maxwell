

#include "AssemblerTELocal.h"

namespace MaxwellProblem::Assembling {
// Serial

void assemble_mass_matrix_cell_H(
	dealii::FullMatrix<double> &cell_matrix,
	const dealii::FEValues<2> &fe_v,
	unsigned int dofs_per_cell_H) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const std::vector<double> &JxW = fe_v.get_JxW_values();

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point)
	for (unsigned int i = 0; i < dofs_per_cell_H; ++i) {
	  const auto H_i = fe_v[H].value(i, q_point);
	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto H_j = fe_v[H].value(j, q_point);

		cell_matrix(i, j) += H_j * H_i * JxW[q_point];
	  }
	}
}

void assemble_cell_curl(
	dealii::FullMatrix<double> &cell_curl_matrix,
	const dealii::FEValues<2> &fe_v,
	const double dofs_per_cell_E,
	const double dofs_per_cell_H) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const dealii::FEValuesExtractors::Vector E(1);

  const auto &JxW = fe_v.get_JxW_values();

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point)
	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_i = fe_v[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto grad_H_j = fe_v[H].gradient(j, q_point);

		// calculate TE curl
		dealii::Tensor<1, 2> curl_H_j;
		curl_H_j[0] = grad_H_j[1];
		curl_H_j[1] = -grad_H_j[0];

		cell_curl_matrix(i, j) += curl_H_j * E_i * JxW[q_point];
	  }
	}
}

void assemble_face_curl(
	const dealii::FEFaceValuesBase<2> &fe_v,
	const dealii::FEFaceValuesBase<2> &fe_v_neighbor,
	dealii::FullMatrix<double> &cell_curl_matrix,
	dealii::FullMatrix<double> &cell_face_matrix_ext_int,
	dealii::FullMatrix<double> &cell_face_matrix_int_ext,
	dealii::FullMatrix<double> &cell_curl_matrix_neighbor,
	double eps,
	double mu,
	double eps_neighbor,
	double mu_neighbor) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const dealii::FEValuesExtractors::Vector E(1);

  double w = std::sqrt(eps / mu);
  double w_ext = std::sqrt(eps_neighbor / mu_neighbor);
  double w_ges = w + w_ext;
  w /= w_ges;
  w_ext /= w_ges;

  const std::vector<double> &JxW = fe_v.get_JxW_values();

  const auto dofs_per_cell_E = cell_curl_matrix.n_rows();
  const auto dofs_per_cell_H = cell_curl_matrix.n_cols();

  const auto &normals = fe_v.get_normal_vectors();

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point) {
	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_i = fe_v[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto H_j = fe_v[H].value(j, q_point);
		const auto n_times_H_j = dealii::cross_product_2d(normals[q_point])*H_j;

		cell_curl_matrix(i, j) -= w * E_i * n_times_H_j * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_i = fe_v[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto H_neighbor_j = fe_v_neighbor[H].value(j, q_point);
		const auto n_times_H_neighbor_j =
			dealii::cross_product_2d(normals[q_point])*H_neighbor_j;

		cell_face_matrix_int_ext(i, j) += w * E_i * n_times_H_neighbor_j * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_neighbor_i = fe_v_neighbor[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {

		const auto H_j = fe_v[H].value(j, q_point);
		const auto n_times_H_j = dealii::cross_product_2d(normals[q_point])*H_j;

		cell_face_matrix_ext_int(i, j) += -w_ext * E_neighbor_i * n_times_H_j * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_neighbor_i = fe_v_neighbor[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {

		const auto H_neighbor_j = fe_v_neighbor[H].value(j, q_point);
		const auto n_times_H_neighbor_j = dealii::cross_product_2d(normals[q_point])*H_neighbor_j;

		cell_curl_matrix_neighbor(i, j) += w_ext * E_neighbor_i * n_times_H_neighbor_j * JxW[q_point];
	  }
	}
  }
}

void assemble_face_stab(
	const dealii::FEFaceValuesBase<2> &fe_v,
	const dealii::FEFaceValuesBase<2> &fe_v_neighbor,
	dealii::FullMatrix<double> &cell_stab_matrix,
	dealii::FullMatrix<double> &cell_face_matrix_ext_int,
	dealii::FullMatrix<double> &cell_face_matrix_int_ext,
	dealii::FullMatrix<double> &cell_stab_matrix_neighbor,
	double eps,
	double mu,
	double eps_neighbor,
	double mu_neighbor) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const dealii::FEValuesExtractors::Vector E(1);

  double w = std::sqrt(eps / mu);
  double w_ext = std::sqrt(eps_neighbor / mu_neighbor);
  double a = 1 / (w + w_ext);
  double b = w * w_ext / (w + w_ext);

  const std::vector<double> &JxW = fe_v.get_JxW_values();
  const auto &normals = fe_v.get_normal_vectors();

  const unsigned int dofs_cell = fe_v.dofs_per_cell;
  const unsigned int dofs_neighbor = fe_v_neighbor.dofs_per_cell;

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point) {
	for (unsigned int i = 0; i < dofs_cell; ++i) {
	  const auto H_i = fe_v[H].value(i, q_point);
	  const auto E_i = fe_v[E].value(i, q_point);
	  const auto n_times_H_i = dealii::cross_product_2d(normals[q_point])*H_i;
	  const auto n_times_E_i = dealii::cross_product_2d(normals[q_point])*E_i;

	  for (unsigned int j = 0; j < dofs_cell; ++j) {
		const auto H_j = fe_v[H].value(j, q_point);
		const auto E_j = fe_v[E].value(j, q_point);
		const auto n_times_H_j = dealii::cross_product_2d(normals[q_point])*H_j;
		const auto n_times_E_j = dealii::cross_product_2d(normals[q_point])*E_j;

		cell_stab_matrix(i, j) +=
			(a * n_times_H_j * n_times_H_i + b * n_times_E_j * n_times_E_i) * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_cell; ++i) {
	  const auto H_i = fe_v[H].value(i, q_point);
	  const auto E_i = fe_v[E].value(i, q_point);
	  const auto n_times_H_i = dealii::cross_product_2d(normals[q_point])*H_i;
	  const auto n_times_E_i = dealii::cross_product_2d(normals[q_point])*E_i;

	  for (unsigned int l = 0; l < dofs_neighbor; ++l) {

		const auto H_neighbor_l = fe_v_neighbor[H].value(l, q_point);
		const auto E_neighbor_l = fe_v_neighbor[E].value(l, q_point);
		const auto n_times_H_neighbor_l = dealii::cross_product_2d(normals[q_point])*H_neighbor_l;
		const auto n_times_E_neighbor_l = dealii::cross_product_2d(normals[q_point])*E_neighbor_l;

		cell_face_matrix_int_ext(i, l) -=
			(a * n_times_H_neighbor_l * n_times_H_i
			 + b * n_times_E_neighbor_l * n_times_E_i)
			* JxW[q_point];
	  }
	}

	for (unsigned int k = 0; k < dofs_neighbor; ++k) {
	  const auto H_neighbor_k = fe_v_neighbor[H].value(k, q_point);
	  const auto E_neighbor_k = fe_v_neighbor[E].value(k, q_point);
	  const auto n_times_H_neighbor_k = dealii::cross_product_2d(normals[q_point])*H_neighbor_k;
	  const auto n_times_E_neighbor_k = dealii::cross_product_2d(normals[q_point])*E_neighbor_k;

	  for (unsigned int j = 0; j < dofs_cell; ++j) {
		const auto H_j = fe_v[H].value(j, q_point);
		const auto E_j = fe_v[E].value(j, q_point);
		const auto n_times_H_j = dealii::cross_product_2d(normals[q_point])*H_j;
		const auto n_times_E_j = dealii::cross_product_2d(normals[q_point])*E_j;

		cell_face_matrix_ext_int(k, j) -=
			(a * n_times_H_j * n_times_H_neighbor_k
			 + b * n_times_E_j * n_times_E_neighbor_k)
			* JxW[q_point];
	  }
	}

	for (unsigned int k = 0; k < dofs_neighbor; ++k) {
	  const auto H_neighbor_k = fe_v_neighbor[H].value(k, q_point);
	  const auto E_neighbor_k = fe_v_neighbor[E].value(k, q_point);
	  const auto n_times_H_neighbor_k = dealii::cross_product_2d(normals[q_point])*H_neighbor_k;
	  const auto n_times_E_neighbor_k = dealii::cross_product_2d(normals[q_point])*E_neighbor_k;

	  for (unsigned int l = 0; l < dofs_neighbor; ++l) {
		const auto H_neighbor_l = fe_v_neighbor[H].value(l, q_point);
		const auto E_neighbor_l = fe_v_neighbor[E].value(l, q_point);
		const auto n_times_H_neighbor_l = dealii::cross_product_2d(normals[q_point])*H_neighbor_l;
		const auto n_times_E_neighbor_l = dealii::cross_product_2d(normals[q_point])*E_neighbor_l;

		cell_stab_matrix_neighbor(k, l) +=
			(a * n_times_H_neighbor_l * n_times_H_neighbor_k
			 + b * n_times_E_neighbor_l * n_times_E_neighbor_k)
			* JxW[q_point];
	  }
	}
  }
}

void assemble_boundary_stab(
	const dealii::FEFaceValues<2> &fe_v,
	dealii::FullMatrix<double> &cell_stab_matrix_E,
	double eps,
	double mu) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const dealii::FEValuesExtractors::Vector E(1);

  double w = std::sqrt(eps / mu);
  const auto JxW = fe_v.get_JxW_values();
  const auto normals = fe_v.get_normal_vectors();

  const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point) {
	for (unsigned int i = 0; i < dofs_per_cell; ++i) {
	  const auto E_i = fe_v[E].value(i, q_point);
	  const auto n_times_E_i = dealii::cross_product_2d(normals[q_point])*E_i;

	  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
		const auto E_j = fe_v[E].value(j, q_point);
		const auto n_times_E_j = dealii::cross_product_2d(normals[q_point])*E_j;
		cell_stab_matrix_E(i, j) += w * n_times_E_j * n_times_E_i * JxW[q_point];
	  }
	}
  }
}

void assemble_rhs_vector_cell(
	dealii::Vector<double> &cell_vector,
	std::vector<dealii::Vector<double>> &rhs_values,
	const dealii::FEValues<2> &fe_v,
	unsigned int dofs_per_cell) {	

	const std::vector<double> &weights = fe_v.get_JxW_values();

	for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point) {
		for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			const unsigned int component = fe_v.get_fe().system_to_component_index(i).first;

			cell_vector(i) += rhs_values[q_point](component) * fe_v.shape_value(i, q_point) * weights[q_point];
		}
	}
}

//Parallel

void mass_cell_worker_H(
	const dealii::DoFHandler<2>::active_cell_iterator &cell,
	MassScratchDataTE &scratch_data,
	MassCopyDataTE &copy_data) {

  scratch_data.fe_values.reinit(cell);
  const dealii::FEValues<2> &fe_v = scratch_data.fe_values;
  const dealii::FEValuesExtractors::Scalar H(0);

  auto dofs_per_cell = fe_v.dofs_per_cell;
  auto dofs_per_cell_H = fe_v.get_fe().get_sub_fe(fe_v.get_fe().component_mask(H)).dofs_per_cell;

  double mu_value = scratch_data.mu.value(cell->center());
  double eps_value = scratch_data.eps.value(cell->center());
  copy_data.reinit(cell, dofs_per_cell, dofs_per_cell_H, mu_value, eps_value);

  assemble_mass_matrix_cell_H(copy_data.cell_matrix, fe_v, dofs_per_cell_H);
}

void mass_cell_worker_inv_H(
	const dealii::DoFHandler<2>::active_cell_iterator &cell,
	MassScratchDataTE &scratch_data,
	MassCopyDataInvTE &copy_data) {

  scratch_data.fe_values.reinit(cell);
  const dealii::FEValues<2> &fe_v = scratch_data.fe_values;
  const dealii::FEValuesExtractors::Scalar H(0);

  auto dofs_per_cell = fe_v.dofs_per_cell;
  auto dofs_per_cell_H = fe_v.get_fe().get_sub_fe(fe_v.get_fe().component_mask(H)).dofs_per_cell;

  double mu_value = scratch_data.mu.value(cell->center());
  double eps_value = scratch_data.eps.value(cell->center());
  copy_data.reinit(cell, dofs_per_cell, dofs_per_cell_H, mu_value, eps_value);

  assemble_mass_matrix_cell_H(copy_data.cell_matrix, fe_v, dofs_per_cell_H);

  copy_data.cell_matrix_inv = copy_data.cell_matrix;
  copy_data.cell_matrix_inv.gauss_jordan();
}

void curl_cell_worker(
	const dealii::DoFHandler<2>::active_cell_iterator &cell,
	CurlScratchDataTE &scratch_data,
	CurlCopyDataTE &copy_data) {

  // get extractors
  dealii::FEValuesExtractors::Scalar H(0);
  dealii::FEValuesExtractors::Vector E(1);

  // get fe_values
  auto &fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  // get dofs per cell
  auto dofs_per_cell_H = fe_values.get_fe()
							 .get_sub_fe(fe_values.get_fe().component_mask(H))
							 .dofs_per_cell;
  auto dofs_per_cell_E = fe_values.get_fe()
							 .get_sub_fe(fe_values.get_fe().component_mask(E))
							 .dofs_per_cell;

  // initialize copy data
  copy_data.reinit(cell, dofs_per_cell_E, dofs_per_cell_H);// This does not reset the data!

  assemble_cell_curl(copy_data.cell_matrix, fe_values, dofs_per_cell_E, dofs_per_cell_H);

  for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<2>::faces_per_cell; ++face_no) {
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

		scratch_data.fe_subface_values.reinit(cell, face_no, subface_no);
		scratch_data.fe_face_values_neighbor.reinit(neighbor_child, neighbor2);

		// get dof for cell an neighbor
		auto &fe_values = scratch_data.fe_values;
		fe_values.reinit(neighbor_child);
		auto neighbor_dofs_per_cell_H = fe_values.get_fe()
											.get_sub_fe(fe_values.get_fe().component_mask(H))
											.dofs_per_cell;
		auto neighbor_dofs_per_cell_E = fe_values.get_fe()
											.get_sub_fe(fe_values.get_fe().component_mask(E))
											.dofs_per_cell;

		copy_data.face_data.emplace_back();
		CurlCopyFaceDataTE &copy_data_face = copy_data.face_data.back();
		copy_data_face.reinit(
			neighbor_child,
			dofs_per_cell_E,
			dofs_per_cell_H,
			neighbor_dofs_per_cell_E,
			neighbor_dofs_per_cell_H);

		auto eps = scratch_data.eps_function.value(cell->center());
		auto mu = scratch_data.mu_function.value(cell->center());
		auto eps_neighbor = scratch_data.eps_function.value(neighbor_child->center());
		auto mu_neighbor = scratch_data.mu_function.value(neighbor_child->center());

		assemble_face_curl(scratch_data.fe_subface_values,
						   scratch_data.fe_face_values_neighbor,
						   copy_data.cell_matrix,
						   copy_data_face.face_matrix_ext_int,
						   copy_data_face.face_matrix_int_ext,
						   copy_data_face.cell_matrix_ext,
						   eps,
						   mu,
						   eps_neighbor,
						   mu_neighbor);
	  }
	} else if (!cell->neighbor_is_coarser(face_no)
			   && (neighbor->index() > cell->index()
				   || (neighbor->level() < cell->level() && neighbor->index() == cell->index()))) {
	  const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

	  scratch_data.fe_face_values.reinit(cell, face_no);
	  scratch_data.fe_face_values_neighbor.reinit(neighbor, neighbor2);

	  // get dof for cell an neighbor
	  auto &fe_values = scratch_data.fe_values;
	  fe_values.reinit(neighbor);
	  auto neighbor_dofs_per_cell_H = fe_values.get_fe()
										  .get_sub_fe(fe_values.get_fe().component_mask(H))
										  .dofs_per_cell;
	  auto neighbor_dofs_per_cell_E = fe_values.get_fe()
										  .get_sub_fe(fe_values.get_fe().component_mask(E))
										  .dofs_per_cell;

	  copy_data.face_data.emplace_back();
	  CurlCopyFaceDataTE &copy_data_face = copy_data.face_data.back();
	  copy_data_face.reinit(
		  neighbor,
		  dofs_per_cell_E,
		  dofs_per_cell_H,
		  neighbor_dofs_per_cell_E,
		  neighbor_dofs_per_cell_H);

	  auto eps = scratch_data.eps_function.value(cell->center());
	  auto mu = scratch_data.mu_function.value(cell->center());
	  auto eps_neighbor = scratch_data.eps_function.value(neighbor->center());
	  auto mu_neighbor = scratch_data.mu_function.value(neighbor->center());

	  assemble_face_curl(scratch_data.fe_face_values,
						 scratch_data.fe_face_values_neighbor,
						 copy_data.cell_matrix,
						 copy_data_face.face_matrix_ext_int,
						 copy_data_face.face_matrix_int_ext,
						 copy_data_face.cell_matrix_ext,
						 eps,
						 mu,
						 eps_neighbor,
						 mu_neighbor);
	}
  }
}

void stab_cell_worker(
	const dealii::DoFHandler<2>::active_cell_iterator &cell,
	StabScratchDataTE &scratch_data,
	StabCopyDataTE &copy_data) {

  auto dofs_per_cell = scratch_data.fe_face_values.get_fe().dofs_per_cell;

  double eps, mu;
  double eps_neighbor, mu_neighbor;

  eps = scratch_data.eps_function.value(cell->center());
  mu = scratch_data.mu_function.value(cell->center());

  copy_data.reinit(cell, dofs_per_cell);

  //Assemble Stab Term (consists only of face terms)
  for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<2>::faces_per_cell; ++face_no) {
	auto face = cell->face(face_no);

	if (face->at_boundary()) {
	  scratch_data.fe_face_values.reinit(cell, face_no);
	  assemble_boundary_stab(scratch_data.fe_face_values, copy_data.cell_matrix, eps, mu);
	} else {
	  Assert(cell->neighbor(face_no).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
	  auto neighbor = cell->neighbor(face_no);
	  if (face->has_children()) {
		const unsigned int neighbor2 = cell->neighbor_face_no(face_no);
		for (unsigned int subface_no = 0; subface_no < face->n_active_descendants(); ++subface_no) {
		  auto neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);
		  Assert(!neighbor_child->has_children(), dealii::ExcInternalError());

		  eps_neighbor = scratch_data.eps_function.value(neighbor_child->center());
		  mu_neighbor = scratch_data.mu_function.value(neighbor_child->center());

		  scratch_data.fe_subface_values.reinit(cell, face_no, subface_no);
		  scratch_data.fe_face_values_neighbor.reinit(neighbor_child, neighbor2);

		  copy_data.face_data.emplace_back();
		  auto &copy_data_face = copy_data.face_data.back();
		  auto neighbor_dofs_per_cell = scratch_data.fe_face_values_neighbor.get_fe().dofs_per_cell;
		  copy_data_face.reinit(neighbor_child, dofs_per_cell, neighbor_dofs_per_cell);

		  assemble_face_stab(scratch_data.fe_subface_values,
							 scratch_data.fe_face_values_neighbor,
							 copy_data.cell_matrix,
							 copy_data_face.face_matrix_ext_int,
							 copy_data_face.face_matrix_int_ext,
							 copy_data_face.cell_matrix_ext,
							 eps,
							 mu,
							 eps_neighbor,
							 mu_neighbor);
		}
	  } else {
		if (!cell->neighbor_is_coarser(face_no) && (neighbor->index() > cell->index() || (neighbor->level() < cell->level() && neighbor->index() == cell->index()))) {
		  const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

		  eps_neighbor = scratch_data.eps_function.value(neighbor->center());
		  mu_neighbor = scratch_data.mu_function.value(neighbor->center());

		  scratch_data.fe_face_values.reinit(cell, face_no);
		  scratch_data.fe_face_values_neighbor.reinit(neighbor, neighbor2);

		  copy_data.face_data.emplace_back();
		  auto &copy_data_face = copy_data.face_data.back();
		  auto neighbor_dofs_per_cell = scratch_data.fe_face_values_neighbor.get_fe().dofs_per_cell;
		  copy_data_face.reinit(neighbor, dofs_per_cell, neighbor_dofs_per_cell);

		  assemble_face_stab(scratch_data.fe_face_values,
							 scratch_data.fe_face_values_neighbor,
							 copy_data.cell_matrix,
							 copy_data_face.face_matrix_ext_int,
							 copy_data_face.face_matrix_int_ext,
							 copy_data_face.cell_matrix_ext,
							 eps,
							 mu,
							 eps_neighbor,
							 mu_neighbor);
		}
	  }
	}
  }
}

void rhs_cell_worker(
	const dealii::DoFHandler<2>::active_cell_iterator &cell,
	RhsScratchDataTE &scratch_data,
	RhsCopyDataTE &copy_data) {

	scratch_data.fe_values.reinit(cell);
	const dealii::FEValues<2> &fe_v = scratch_data.fe_values;

	auto dofs_per_cell = fe_v.dofs_per_cell;

	const unsigned int n_q_points = fe_v.n_quadrature_points;
	const unsigned int n_components_fe = fe_v.get_fe().n_components();
	std::vector<dealii::Vector<double>> rhs_values(
		n_q_points, dealii::Vector<double>(n_components_fe));

	scratch_data.rhs_function.vector_value_list(fe_v.get_quadrature_points(), rhs_values);

	copy_data.reinit(cell, dofs_per_cell);

	assemble_rhs_vector_cell(copy_data.cell_vector, rhs_values, fe_v, dofs_per_cell);
}

}// namespace MaxwellProblem::Assembling